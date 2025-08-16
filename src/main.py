import torch
from models.LLM import LLM

if __name__=="__main__":
	device = "cpu" ## "cuda" if torch.cuda.is_available() else "cpu"
	new_model = LLM(llm_model_name="Qwen/Qwen2.5-1.5B-Instruct", device=device, max_new_tokens=512)
    
	while True:
		# Conteúdo digitado pelo usuário
		user_message = input(f"Digite o que deseja saber:\n") # "Qual seu nome?"
		if user_message=="bye":
			break
		print(f"Resposta do modelo: {new_model.llm_response(user_message)}. Digite 'bye' para finalizar a conversa.")