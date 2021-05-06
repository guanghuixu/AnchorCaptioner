# Copyright (c) Facebook, Inc. and its affiliates.
from pythia.common.registry import registry
from pythia.models.m4c import M4C


@registry.register_model("m4c_captioner")
class M4CCaptioner(M4C):
    def __init__(self, config):
        super().__init__(config)
        self.remove_unk_in_pred = self.config.remove_unk_in_pred

    def _forward_output(self, sample_list, fwd_results):
        super()._forward_output(sample_list, fwd_results)

        if (not self.training) and self.remove_unk_in_pred:
            # avoid outputting <unk> in the generated captions
            fwd_results["scores"][..., self.answer_processor.UNK_IDX] = -1e10

        return fwd_results
