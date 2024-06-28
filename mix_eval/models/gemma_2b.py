from mix_eval.models.base import BaseModel
from mix_eval.api.registry import register_model

import torch

from transformers import AutoModelForCausalLM
from mix_eval.utils.common_utils import get_gpu_memory

@register_model("gemma_2b")
class Gemma_2B(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.model_name = args.model_path
        self.attn_implementation = 'flash_attention_2' # If use default, set to None

        self.model = self.build_model()
        self.model_max_len = self.model.config.max_position_embeddings 
        self.tokenizer = self.build_tokenizer()
        self.max_input_length_closeend = 2048
        
    def build_model(self):
        num_gpus = torch.cuda.device_count()
        kwargs = {}
        kwargs["device_map"] = "auto"
        if self.args.max_gpu_memory is None:
            kwargs[
                "device_map"
            ] = "sequential"  # This is important for not the same VRAM sizes
            available_gpu_memory = get_gpu_memory(num_gpus)
            kwargs["max_memory"] = {
                i: str(int(available_gpu_memory[i] * 0.85)) + "GiB"
                for i in range(num_gpus)
            }
        else:
            kwargs["max_memory"] = {i: self.args.max_gpu_memory for i in range(num_gpus)}
        
        if self.attn_implementation is not None:
            kwargs["attn_implementation"] = self.attn_implementation
            
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.model_dtype,
            trust_remote_code=self.trust_remote_code,
            **kwargs
        ).eval()
        
        return model