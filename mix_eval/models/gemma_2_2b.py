import torch
import json

from mix_eval.models.base import ChatModel
from mix_eval.api.registry import register_model

@register_model("gemma_2_2b")
class Gemma_2_2B(ChatModel):
    def __init__(self, args):
        super().__init__(args)
        self.model_name = args.model_path
        self.attn_implementation = 'flash_attention_2' # If use default, set to None
        self.model_dtype = torch.bfloat16
        
        self.SYSTEM_MESSAGE = None # set to None if no system message
        self.USER_MESSAGE_TEMPLATE = lambda x: {"role": "user", "content": x}
        self.ASSISTANT_MESSAGE_TEMPLATE = lambda x: {"role": "assistant", "content": x}
        
        self.model = self.build_model()
        self.model_max_len = self.model.config.max_position_embeddings
        self.tokenizer = self.build_tokenizer()
        self.max_input_length_closeend = min(
            self.model_max_len,
            self.max_input_length
        ) - self.closeended_max_new_tokens
        self.max_input_length_openend = min(
            self.model_max_len,
            self.max_input_length
        ) - self.openended_max_new_tokens
    
    def apply_chat_template(self, messages):
        '''
        If the tokenizer has a chat template, apply it to the messages.
        If not, implement a custom one here.
        '''
        
        # Custom template logic if tokenizer does not support chat templates
        formatted_messages = []
        
        for message in messages:
            if message['role'] == 'user':
                formatted_message = f"<start_of_turn>user\n{message['content']}\n<end_of_turn>"
            elif message['role'] == 'system' or message['role'] == 'model':
                formatted_message = f"<start_of_turn>model\n{message['content']}\n<end_of_turn>"
            else:
                formatted_message = message['content']  # handle other roles if necessary
            formatted_messages.append(formatted_message)
        
        # Concatenate all formatted messages into one string
        return "\n".join(formatted_messages)
        
    