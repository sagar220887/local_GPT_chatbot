import streamlit as st

from src.data_ingestion import *

from constants import *
from src.embeddings import *
from src.vector_db import *
from src.llm_model import *
from src.prompt_template import *
from src.chain import *
from src.events import *

from dotenv import load_dotenv
load_dotenv()

HF_API_KEY = os.getenv('HUGGINGFACE_API_KEY')



if "processComplete" not in st.session_state:
    st.session_state.processComplete = None
if "new_data_source" not in st.session_state:
    st.session_state.new_data_source = False
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = None

if "data_chunks" not in st.session_state:
    st.session_state.data_chunks = None

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# file process button state
if "file_process" not in st.session_state:
    st.session_state.file_process = None

# Initialize chat history
if "upload_files" not in st.session_state:
    st.session_state.upload_files = None




st.markdown(
    """
<style>
    .st-emotion-cache-4oy321 {
        flex-direction: row-reverse;
        text-align: right;
    }
</style>
""",
    unsafe_allow_html=True,
)



#########################################################################################################

def get_streamlit_instance():
    return st



#######################################################################################################



def main():
    st.title("Chat with Your documents !!")

    ############# SIDE NAVIGATION BAR #######################################
    with st.sidebar:
        st.title("Data sources")
        st.session_state.upload_files =  st.file_uploader(
            "Upload your file",
            type=['pdf','txt', 'csv', 'doc/docx'],
            accept_multiple_files=True
        )
        URL_TO_EXTRACT = st.text_input("Site URL")
        st.session_state.file_process = st.button("Process")

        if st.session_state.file_process:
            if st.session_state.upload_files:
                st.session_state.new_data_source = True
                # Getting the chain
                st.session_state.conversation = process_for_new_data_source(st.session_state.upload_files, None)

            elif URL_TO_EXTRACT:
                st.session_state.new_data_source = True
                st.session_state.conversation = process_for_new_data_source(None, URL_TO_EXTRACT)
                

        else :
            # As no file process required 
            st.session_state.conversation = process_for_existing_source()




    ##### CHAT BOT CONVERSATIONS  ############################
            
    with st.chat_message("assistant"):
        st.write("Hello Human, How may I help you.")

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        print('message - ', message)
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask Question about your files."):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        print('st.session_state.new_data_source ---> ', st.session_state.new_data_source)
        if st.session_state.new_data_source == False:
            process_for_existing_source()

        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            print('Inside st.chat_message("assistant")')
            with st.spinner('Processing ...'):
                response = get_response(st.session_state.conversation, prompt)
                st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        


if __name__ == '__main__':
    main()