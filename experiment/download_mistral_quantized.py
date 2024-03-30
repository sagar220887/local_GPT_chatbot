import os
from langchain.llms import CTransformers
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from ctransformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from langchain.llms  import LlamaCpp
from langchain.callbacks.manager import CallbackManager


model_id = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
model_file = "./model/mistral-7b-instruct-v0.1.Q4_K_S.gguf" # Downloaded and placed in local directory

llama2_model_path = "model/llama-2-7b-chat.ggmlv3.q8_0.bin"
config={'max_new_tokens':128,
                    'temperature':0.01}


llm=CTransformers(
        model=llama2_model_path,
        model_type="llama",
        config={'max_new_tokens':128,
                'temperature':0.01}
)

text = "<s>[INST] What is your favourite condiment? [/INST]"
"Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!</s> "
"[INST] Do you have mayonnaise recipes? [/INST]"


response  = llm(text)
print(response)