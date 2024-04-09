
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
from src.helper import *

st = get_streamlit_instance()



def process_for_existing_source():
    with st.spinner('Processing, Wait for it...'):
        print("Processing for existing source ======>>")

        #Load the Embedding Model
        embeddings = create_embeddings()
        st.session_state.vector_db = load_vectordb(VECTOR_DB_DIRECTORY, embeddings=embeddings)
        # llm = get_llm_model()
        st.session_state.llm_model = get_user_input_llm_model(st.session_state.llmprovider, st.session_state.apikey)
        qa_prompt = get_qa_prompt()
        st.session_state.conversation = create_chain(st.session_state.llm_model, st.session_state.vector_db, qa_prompt)
        return st.session_state.conversation



def get_response(user_query):
    print("<< ============== Getting the response of the User Question ======>>")
    print('user_query - ', user_query)
    # check the instance of chain
    print("st.session_state.conversation ==> ", st.session_state.conversation)
    if st.session_state.conversation == None:
        process_for_existing_source()

    # Getting the similarity
    vector_db_results = get_similiar_docs(st.session_state.vector_db, user_query, k=2, score=True)
    print('vector_db_results ==> ', vector_db_results)
    # return "\n".join(vector_db_results)

    # executing Query through chain
    result=st.session_state.conversation({'query':user_query}, return_only_outputs=True)
    print('result - ', result)
    ans = result['result']
    print(f"Answer:{ans}")
    return ans



def process_for_new_data_source(uploaded_files, web_url):

    with st.spinner('Processing, Wait for it...'):
            
        print('uploaded_files - ', uploaded_files)
        print('web_url - ', web_url)
        
        if uploaded_files:
            # #Load the PDF File
            documents = load_data_source(uploaded_files)
        elif web_url:
            # Extracting text from web page
            documents = extract_data_from_webpage(web_url)

        # #Split Text into Chunks
        st.session_state.data_chunks = get_data_chunks(documents)

        # #Load the Embedding Model
        embeddings = create_embeddings()

        # #Convert the Text Chunks into Embeddings and Create a FAISS Vector Store
        st.session_state.vector_db=store_data_in_vectordb(st.session_state.data_chunks, embeddings)

        ## Loading the LLM model
        # llm = get_llm_model()
        st.session_state.llm_model = get_user_input_llm_model(st.session_state.llmprovider, st.session_state.apikey)
        
        ## Getting the prompt
        qa_prompt = get_qa_prompt()
        
        ## Getting the conversation chain
        st.session_state.conversation = create_chain(st.session_state.llm_model, st.session_state.vector_db, qa_prompt)
        # st.write('✅ Created Chain')

        st.text("Ready to go ...✅✅✅")
        st.session_state.processComplete = True

        return st.session_state.conversation
        

def update_chain_on_key_selection(llm, api_key):
    with st.spinner('Processing, Wait for it...'):
        print('<< ===  Updating the llm model and chain on key selection == >>')
        st.session_state.llm_model = get_user_input_llm_model(llm, api_key)
        qa_prompt = get_qa_prompt()
        if st.session_state.vector_db:
            st.session_state.conversation = create_chain(st.session_state.llm_model, st.session_state.vector_db, qa_prompt)
        else:
            print('Vector DB not created')
        st.write('✅ Token updated')
        return st.session_state.conversation

