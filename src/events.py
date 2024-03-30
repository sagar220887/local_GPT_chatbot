
from src.data_ingestion import (
    load_data_source,
    extract_data_from_webpage,
    get_data_chunks
) 

from constants import *
from src.embeddings import *
from src.vector_db import *
from src.llm_model import *
from src.prompt_template import *
from src.chain import *


def process_for_existing_source():

    #Load the Embedding Model
    embeddings = create_embeddings()
    vector_db = load_vectordb(VECTOR_DB_DIRECTORY, embeddings=embeddings)
    llm = get_llm_model()
    qa_prompt = get_qa_prompt()
    chain = create_chain(llm, vector_db, qa_prompt)
    return chain



def get_response(chain, user_query):
    print('user_query - ', user_query)
    # executing Query through chain
    result=chain({'query':user_query}, return_only_outputs=True)
    print('result - ', result)
    ans = result['result']
    print(f"Answer:{ans}")
    return ans