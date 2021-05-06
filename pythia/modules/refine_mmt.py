import torch
from torch import nn
import torch.nn.functional as F
import functools

from pytorch_transformers.modeling_bert import (
    BertLayerNorm, BertEmbeddings, BertEncoder, BertConfig,
    BertPreTrainedModel
)

class Refine_MMT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.prev_pred_embeddings = PositionEmbeddings(config)
        self.encoder = BertEncoder(config)
        # self.apply(self.init_weights)  # old versions of pytorch_transformers
        self.init_weights()

    def forward(self, visual_emb, visual_mask, anchor_emb, ocr_emb, ocr_mask, dec_emb):

        dec_emb = self.prev_pred_embeddings(dec_emb)

        anchor_mask = torch.ones(anchor_emb.size(0), 1, dtype=torch.float32, device=anchor_emb.device)
        dec_mask = torch.zeros(
            dec_emb.size(0),
            dec_emb.size(1),
            dtype=torch.float32,
            device=dec_emb.device
        )
        encoder_inputs = torch.cat([visual_emb, anchor_emb, ocr_emb, dec_emb], dim=1)
        attention_mask = torch.cat([visual_mask, anchor_mask, ocr_mask, dec_mask], dim=1)
        
        ocr_start = visual_mask.size(1) + 1
        ocr_max_num = ocr_mask.size(-1)
        dec_max_num = dec_mask.size(-1)

        to_seq_length = attention_mask.size(1)
        from_seq_length = to_seq_length

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.repeat(
            1, 1, from_seq_length, 1
        )

        # we don't need language mask
        extended_attention_mask[:, :, -dec_max_num:, -dec_max_num:] = \
            _get_causal_mask(dec_max_num, encoder_inputs.device)

        # flip the mask, so that invalid attention pairs have -10000.
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            encoder_inputs,
            extended_attention_mask,
            head_mask=head_mask
        )

        mmt_seq_output = encoder_outputs[0]
        mmt_ocr_output = mmt_seq_output[:, ocr_start: ocr_start+ocr_max_num]
        mmt_dec_output = mmt_seq_output[:, -dec_max_num:]

        return mmt_ocr_output, mmt_dec_output


class PositionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        MAX_DEC_LENGTH = 100
        hidden_size = config.hidden_size
        ln_eps = config.layer_norm_eps

        self.position_embeddings = nn.Embedding(MAX_DEC_LENGTH, hidden_size)

        self.ans_layer_norm = BertLayerNorm(hidden_size, eps=ln_eps)
        self.emb_layer_norm = BertLayerNorm(hidden_size, eps=ln_eps)
        self.emb_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, raw_dec_emb):
        batch_size, seq_length, _ = raw_dec_emb.size()
        dec_emb = self.ans_layer_norm(raw_dec_emb)

        position_ids = torch.arange(
            seq_length,
            dtype=torch.long,
            device=dec_emb.device
        )
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = self.emb_layer_norm(position_embeddings)
        embeddings = self.emb_dropout(embeddings)
        dec_emb = dec_emb + embeddings

        return dec_emb

@functools.lru_cache(maxsize=32)
def _get_causal_mask(seq_length, device):
    # generate a lower triangular mask
    mask = torch.zeros(seq_length, seq_length, device=device)
    for i in range(seq_length):
        for j in range(i+1):
            mask[i, j] = 1.
    return mask
