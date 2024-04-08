from langchain import HuggingFaceHub
from ctransformers import AutoModelForCausalLM
import os
from constants import *
from langchain.llms import CTransformers

from dotenv import load_dotenv
load_dotenv()

HF_API_KEY = os.getenv('HUGGINGFACE_API_KEY')


def get_user_input_llm_model(api_provider, api_key):
    print("Loading the LLM Model...")
    print("api_provider :: " + api_provider)
    print("api_key :: " + api_key)

    print("Using Local LLM model")
    llm=CTransformers(
        model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
        model_type="llama",
        config={'max_new_tokens':128,
                'temperature':0.01}
    )
    return llm

    # if api_provider == 'Huggingface':
    #     print('Using HUGGINGFACE_API_KEY')
    #     llm = HuggingFaceHub(
    #         repo_id=MODEL_REPO_ID, 
    #         huggingfacehub_api_token=api_key
    #     )
    #     return llm
    # elif api_provider == 'Google':
    #     return None
    
    # elif api_provider == 'OpenAI':
    #     return None
    
    # elif api_provider == 'Local':
    #     print("Using Local LLM model")
    #     llm=CTransformers(
    #         model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
    #         model_type="llama",
    #         config={'max_new_tokens':128,
    #                 'temperature':0.01}
    #     )
    #     return llm
    
    # else:
    #     print('Using DEFAULT HUGGINGFACE_API_KEY')
    #     llm = HuggingFaceHub(
    #         repo_id=MODEL_REPO_ID, 
    #         huggingfacehub_api_token=HF_API_KEY
    #     )
    #     return llm



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
