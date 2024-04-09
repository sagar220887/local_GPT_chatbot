from src.helper import *
from src.events import *


st = get_streamlit_instance()

def left_navigation_layout():
    # st.sidebar.title("Data Sources")
    # st.sidebar.markdown("""
    # <style>
    #     .sidebar-nav {
    #         padding: 9px 0;
    #     }
    #     """)
    

    ############# SIDE NAVIGATION BAR #######################################
    with st.sidebar:
        st.subheader("Data sources", divider="rainbow")
        st.session_state.upload_files =  st.file_uploader(
            "Upload your file",
            type=['pdf','txt', 'csv', 'doc/docx', 'xlsx', 'xls', 'ppt', 'pptx'],
            accept_multiple_files=True
        )
        URL_TO_EXTRACT = st.text_input("Site URL")
        
        with st.form(key="api_tokens"):

            c1, c2 = st.columns(2)
            with c1:
                st.session_state.llmprovider = st.selectbox("API KEY", ( "Local", "Default","Huggingface", "Google","OpenAI"), index=0)
            with c2:
                st.session_state.apikey = st.text_input("Enter the key", type="password")

            st.session_state.setkey = st.form_submit_button(label='set')

        if st.session_state.setkey:
            st.session_state.conversation = update_chain_on_key_selection(st.session_state.llmprovider, st.session_state.apikey)




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
            # st.session_state.conversation = process_for_existing_source()
            pass


def main_container_layout():
   
    st.subheader("Chat with Your documents !!", divider="rainbow")

  

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

        ## TODO
        # print('st.session_state.new_data_source ---> ', st.session_state.new_data_source)
        # if st.session_state.new_data_source == False:
        #     process_for_existing_source()

        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            print('Inside st.chat_message("assistant")')
            with st.spinner('Processing ...'):
                response = get_response(prompt)
                st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

        
        
















        
        # chat_tab, view_tab = st.tabs([":robot_face: chat with Bot", "View Data"])

        # with chat_tab:

        # css = '''
        # <style>
        #     .stTabs [data-baseweb="tab-list"] {
        #         gap: 15px;
        #     }
        #     .stTabs [data-baseweb="tab-highlight"] {
        #         background-color:teal;
        #     }
        #     button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {font-size: 16px;}
        # </style>
        # '''

        # st.markdown(css, unsafe_allow_html=True)