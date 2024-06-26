from dotenv import load_dotenv

from openai import OpenAI

from mix_eval.models.base_api import APIModelBase
from mix_eval.api.registry import register_model

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

@register_model("custom")
class Custom(APIModelBase):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.model_name = 'custom'
        
        load_dotenv()
        self.client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )