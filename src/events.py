
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
                vector_db=store_data_in_vectordb(st.session_state.data_chunks, embeddings)

                ## Loading the LLM model
                llm = get_llm_model()
                
                ## Getting the prompt
                qa_prompt = get_qa_prompt()
                
                ## Getting the conversation chain
                st.session_state.conversation = create_chain(llm, vector_db, qa_prompt)
                # st.write('✅ Created Chain')

                st.text("Ready to go ...✅✅✅")
                st.session_state.processComplete = True

                return st.session_state.conversation
        


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


