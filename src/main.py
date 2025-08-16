import torch
from models.LLM import LLM

if __name__=="__main__":
	device = "cpu" ## "cuda" if torch.cuda.is_available() else "cpu"
	new_model = LLM(llm_model_name="Qwen/Qwen2.5-1.5B-Instruct", device=device, max_new_tokens=512)
    
	# Conteúdo digitado pelo usuário
	user_message = "Qual seu nome?"
	print(f"Resposta do modelo: {new_model.llm_response(user_message)}")