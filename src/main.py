import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
import torch
from models.LLM import LLM

if __name__=="__main__":
	device = "cuda" if torch.cuda.is_available() else "cpu"
	new_model = LLM(llm_model_name="Qwen/Qwen2.5-0.5B-Instruct", device=device, max_new_tokens=512)
	print("Digite 'bye' para finalizar a conversa.")
	user_message_history = []
	while True:
		# Conteúdo digitado pelo usuário
		user_message = input(f"Digite o que deseja saber:\n") # "Qual seu nome?"
		if user_message=="bye":
			break
		user_message_history.append(user_message+f"Resposta do modelo: \n{new_model.llm_response(user_message)}")
		print(f"Resposta do modelo: \n{user_message_history[-1]}")