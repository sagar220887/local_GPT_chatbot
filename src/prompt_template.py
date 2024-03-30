from langchain import PromptTemplate
from langchain import LLMChain
from langchain.llms import CTransformers
from src.helper import *

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


DEFAULT_SYSTEM_PROMPT="""\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. 
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. 
Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of 
answering something not correct. If you don't know the answer to a question,
please don't share false information."""



#CUSTOM_SYSTEM_PROMPT="You are an advanced assistant that provides translation from English to Hindi"
# CUSTOM_SYSTEM_PROMPT="You are an advanced assistant that provides summarization given any book name"




qa_template="""Use the following pieces of information to answer the user's question.
If you dont know the answer just say you know, don't try to make up an answer.

Context:{context}
Question:{question}

Only return the helpful answer below and nothing else
Helpful answer
"""

translation_instruction = "Convert the following text from English to Hindi: \n\n {text}"
summarization_instruction = "Give a proper summary of the of : \n\n {text}"


def get_summarization_template():
    SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS
    template = B_INST + SYSTEM_PROMPT + summarization_instruction + E_INST
    prompt = PromptTemplate(template=template, input_variables=["text"])

def get_translation_template():
    SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS
    template = B_INST + SYSTEM_PROMPT + translation_instruction + E_INST
    prompt = PromptTemplate(template=template, input_variables=["text"])



def get_qa_prompt():
    SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS
    template = B_INST + SYSTEM_PROMPT + qa_template + E_INST
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return prompt





# def get_qa_prompt():
#     template="""Use the following pieces of information to answer the user's question.
#             If you dont know the answer just say you dont know, don't try to make up an answer.

#             Context:{context}
#             Question:{question}

#             Only return the helpful answer below and nothing else
#             Helpful answer
#             """
    

#     prompt = PromptTemplate(template=template, input_variables=["context", "question"])
#     print('Prompt created')
#     return prompt