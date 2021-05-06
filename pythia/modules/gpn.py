import torch
import torch.nn as nn
import torch.nn.functional as F 

from pytorch_transformers.modeling_bert import (
    BertLayerNorm, BertEmbeddings, BertEncoder, BertConfig,
    BertPreTrainedModel
)

class GraphProposalNetwork(nn.Module):
    def __init__(self, config, image_encoder=None):
        super().__init__()
        # self.v2t = Visual2Text(config.hidden_size, config.num_hidden_layers, config.num_hidden_layers, image_encoder)
        self.t2t = Text2Text(config)
        self.anchor_fc = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ELU(),
            nn.Linear(config.hidden_size // 2, 1)
            )
        self.graph_fc = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ELU(),
            nn.Linear(config.hidden_size // 2, 1),
            nn.Sigmoid()
            )
        self.rnn = nn.GRU(config.hidden_size, config.hidden_size, batch_first=True)

    def forward_rnn(self, text_emb, text_mask, anchor):
        sort_v, sort_id = torch.sort(text_mask.sum(1).clamp_min(1), descending=True)
        recover_id = sort_id.argsort()

        text_emb = text_emb[sort_id]
        anchor = anchor[sort_id]
        text_sort = nn.utils.rnn.pack_padded_sequence(text_emb, sort_v.data.cpu().numpy().tolist(), batch_first=True)
        text_emb, anchor = self.rnn(text_sort, anchor.transpose(0, 1))
        output, _ = nn.utils.rnn.pad_packed_sequence(text_emb, batch_first=True)
        text_emb = output[recover_id]
        anchor = anchor.transpose(0,1)[recover_id]
        return text_emb, anchor


    def forward(self, text_emb, text_mask, visual_emb, visual_mask):
        # hidden_text = self.v2t(text_emb, visual_emb, cover_map, 
        #     padding_mask=text_mask==0, visual_padding=visual_mask==0)
        update_text = self.t2t(torch.cat([visual_emb, text_emb], dim=1), 
        torch.cat([visual_mask, text_mask], dim=1))
        visual_emb = visual_emb + torch.tanh(update_text[:, :visual_mask.size(1)])
        text_emb = text_emb + torch.tanh(update_text[:, -text_mask.size(1):])  # [B, 50, 768]
        anchor_scores = self.anchor_fc(text_emb).squeeze(2)

        argmax_id = anchor_scores.argmax(dim=1)
        anchor = [text_emb[_,argmax_id[_]] for _ in range(argmax_id.size(0))]
        anchor = torch.stack(anchor).unsqueeze(1)
        text_emb_, anchor = self.forward_rnn(text_emb, text_mask, anchor)  # [B, 24, 768]
        
        PAD_size = text_emb_.size(1)
        text_emb = torch.cat([text_emb_, text_emb[:, PAD_size:]], dim=1)
        graph_scores = self.graph_fc(text_emb).squeeze(2) - 0.5
        return visual_emb, text_emb, anchor_scores, graph_scores, anchor


class Visual2Text(nn.Module):
    def __init__(self, d_model=768, nhead=12,  num_layers=4, image_encoder=None):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        if image_encoder:
            self.image_encoder1 = image_encoder
        else:
            self.image_encoder1 = nn.Sequential(
                nn.Linear(2048, 2048),
                nn.ELU(inplace=True)
            )
        self.image_encoder2 = nn.Sequential(
            nn.Linear(2048, d_model),
            BertLayerNorm(d_model)
        )
        self.update_gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ELU(inplace=True),
        )

    def forward(self, text_emb, visual_emb, cover_map, padding_mask, visual_padding):
        """
        text_emb: [B, N, d]
        visual_emb: [B, 14*14, d]
        cover_map: List(B, N), [[[0,1], [4]], ...]]
        """
        # visual_emb = self.image_encoder1(visual_emb)
        # visual_emb = F.normalize(visual_emb, dim=-1)
        # visual_emb = self.image_encoder2(visual_emb)
        # hidden_node_emb = []
        # for b in range(text_emb.size(0)):
        #     cover_ids = cover_map[b]
        #     hidden_t = []
        #     for i, cover_id in enumerate(cover_ids):
        #         t = text_emb[b][i].unsqueeze(0).unsqueeze(0)  # [1, 1, d]
        #         v = visual_emb[b][cover_id].unsqueeze(1)  # [M, 1, d]
        #         t_ = self.transformer(t, v)  # [1, 1, d]
        #         hidden_t.append(t.squeeze())
        #     num_t = len(hidden_t)
        #     if num_t:
        #         hidden_t = torch.stack(hidden_t, dim=0)
        #         hidden_node_emb.append(torch.cat([hidden_t, text_emb[b, num_t:]], dim=0))
        #     else:
        #         hidden_node_emb.append(text_emb[b])
        # hidden_node_emb = torch.stack(hidden_node_emb, dim=0)  # [B, N, d]
        # text_emb = self.update_gate(torch.cat([text_emb, hidden_node_emb], dim=2))
        text_emb_ = self.transformer(text_emb.transpose(0, 1), 
            visual_emb.transpose(0, 1), 
            tgt_key_padding_mask=padding_mask,
            memory_key_padding_mask=visual_padding
            )
        # text_emb = text_emb + torch.tanh(text_emb_.transpose(0, 1))
        return text_emb


class Text2Text(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.encoder = BertEncoder(config)
        # self.apply(self.init_weights)  # old versions of pytorch_transformers
        self.init_weights()

    def forward(self, encoder_inputs,  attention_mask):
        # generate the attention mask similar to prefix LM
        # all elements can attend to the elements in encoding steps
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.repeat(
            1, 1, attention_mask.size(1), 1
        )

        # flip the mask, so that invalid attention pairs have -10000.
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0  # 1 can attention
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            encoder_inputs,
            extended_attention_mask,
            head_mask=head_mask
        )
        return encoder_outputs[0]
    

if __name__ == "__main__":
    text_emb = torch.randn([3, 10, 768])
    visual_emb = torch.randn([3, 14 * 14, 2048])
    cover_map = [[[0,1], [2,5]], [[7,8,9]], [[1], [2],[3]]]
    text_mask = torch.ones([3, 10])
    # v2t = Visual2Text()
    # new_text_emb = v2t(text_emb, visual_emb, cover_map)
    # print(new_text_emb.size())

    # t2t = Text2Text(BertConfig(hidden_size=768))
    # new_text_emb = t2t(new_text_emb, text_mask)

    model = GraphProposalNetwork()
    new_text_emb, scores = model(text_emb, text_mask, visual_emb, cover_map)
    print(new_text_emb.size(), scores)



