import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LLM():
    def __init__(self, llm_model_name:str="Qwen/Qwen2.5-0.5B", max_new_tokens:int=10, device:str="cpu"):        
        self.max_new_tokens=max_new_tokens
        self.device = device
        self.model_name = llm_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
    
    def llm_response(self, user_message:str):
        messages = [
            {"role": "system", "content": "You are a virtual assistant called Fogos' AI assyst"},
            {"role": "user", "content": user_message},
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        # Generate the response
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=self.max_new_tokens,
            do_sample=True, # Use sampling for more varied responses
            top_p=0.7,      # Use top_p to control randomness
            temperature=0.6 # A lower temperature makes the output more focused
        )

        input_token_len = inputs["input_ids"].shape[-1]
        generated_tokens = outputs[0][input_token_len:]

        # Decode only the new tokens, skipping special tokens
        cleaned_response = self.tokenizer.decode(
            generated_tokens, 
            skip_special_tokens=True
        )

        return cleaned_response
