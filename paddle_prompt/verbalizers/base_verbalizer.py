from __future__
from __future__ import annotations

class Verbalizer:
    def __init__(self) -> None:
        self.trigger_mask = None
        
    def replace_trigger_tokens(self,model_inputs, trigger_ids, trigger_mask):
        """Replaces the trigger tokens in input_ids."""
        out = model_inputs.copy()
        input_ids = model_inputs['input_ids']
        trigger_ids = trigger_ids.repeat(trigger_mask.size(0), 1)
        try:
            filled = input_ids.masked_scatter(trigger_mask, trigger_ids)
        except RuntimeError:
            filled = input_ids
        out['input_ids'] = filled
        return out
