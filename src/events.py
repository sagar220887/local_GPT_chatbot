
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
# from main import *



def process_for_existing_source():
    print("Processing for existing source ======>>")

    #Load the Embedding Model
    embeddings = create_embeddings()
    vector_db = load_vectordb(VECTOR_DB_DIRECTORY, embeddings=embeddings)
    # llm = get_llm_model()
    llm = get_user_input_llm_model(st.session_state.llmprovider, st.session_state.apikey)
    qa_prompt = get_qa_prompt()
    st.session_state.conversation = create_chain(llm, vector_db, qa_prompt)
    return st.session_state.conversation



def get_response(chain, user_query):
    print("<< ============== Getting the response of the User Question ======>>")
    print('user_query - ', user_query)
    # check the instance of chain
    print("st.session_state.conversation ==> ", st.session_state.conversation)
    if st.session_state.conversation == None:
        process_for_existing_source()

    # executing Query through chain
    result=st.session_state.conversation({'query':user_query}, return_only_outputs=True)
    print('result - ', result)
    ans = result['result']
    print(f"Answer:{ans}")
    return ans


