from src.ui_layout import *
from src.helper import *
from src.events import *
from src.prompt_template import *
from src.embeddings import *
from src.data_ingestion import *
from src.llm_model import *
from src.vector_db import *


st = get_streamlit_instance()

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


# Initialize llmprovider
if "llmprovider" not in st.session_state:
    st.session_state.llmprovider = None

# Initialize apikey
if "apikey" not in st.session_state:
    st.session_state.apikey = None


# Initialize setkey
if "setkey" not in st.session_state:
    st.session_state.setkey = None

# Initialize llm_model
if "llm_model" not in st.session_state:
    st.session_state.llm_model = None

# Initialize vector db
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# Initialize state of knowledgebase checkbox
if "user_response_knowledgebase" not in st.session_state:
    st.session_state.user_response_knowledgebase = None


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


def main():
    
    main_container_layout()
    left_navigation_layout()



if __name__ == '__main__':
    main()