from langchain import HuggingFaceHub
from ctransformers import AutoModelForCausalLM
import os
from constants import *
from langchain.llms import CTransformers

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

    # llm = AutoModelForCausalLM.from_pretrained(
    #     model_path_or_repo_id="model/", 
    #     model_file = "mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    #     model_type="mistral",
    #     local_files_only = True
    # )
    llm = HuggingFaceHub(
            repo_id=MODEL_REPO_ID, 
            huggingfacehub_api_token=HF_API_KEY
        )
    print('LLM model Loaded')
    return llm
