from langchain import HuggingFaceHub
from transformers import  AutoTokenizer
from ctransformers import AutoModelForCausalLM
import os

from dotenv import load_dotenv
load_dotenv()

HF_API_KEY = os.getenv('HUGGINGFACE_API_KEY')


def get_llm_model():
    # llm=CTransformers(
    #         model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
    #         model_type="llama",
    #         config={'max_new_tokens':128,
    #                 'temperature':0.01}
    # )

    # llm = AutoModelForCausalLM.from_pretrained("./model/mistral-7b-instruct-v0.1.Q4_K_S.gguf", model_type="cpu")
    llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.2", huggingfacehub_api_token=HF_API_KEY)
    print('LLM model Loaded')
    return llm
