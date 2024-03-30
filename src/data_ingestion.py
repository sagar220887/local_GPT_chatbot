import os
from langchain_community.document_loaders import UnstructuredURLLoader, MergedDataLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
import logging

from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from src.prompt_template import *

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s]: %(message)s:'
)


def load_data_source(loaded_files):
    for loaded_file in loaded_files:
        print('loaded_file - ', loaded_file)
        temp_file = create_temp_file(loaded_file)
        # temp_file = './tmp/74862151_1709607183894.pdf'
        # loader = PyPDFLoader(temp_file)
        loader = get_loader_by_file_extension(temp_file)
        print('loader - ', loader)
        data = loader.load()
        return data
    



def extract_data_from_webpage(web_url):
    # load data
    # ! pip3 install unstructured libmagic python-magic python-magic-bin
    loader = UnstructuredURLLoader(urls=web_url)
    documents = loader.load()
    return documents

def get_data_chunks(data):
    recursive_char_text_splitter=RecursiveCharacterTextSplitter(
                                                chunk_size=500,
                                                chunk_overlap=50)
    documents=recursive_char_text_splitter.split_documents(data)
    # print('documents - ', documents)
    print('documents type - ', type(documents))
    print('documents length - ', len(documents))
    return documents




####################################       FUNCTIONS              ############################################


def create_temp_file(loaded_file):
    # save the file temporarily
    temp_file = f"./tmp/{loaded_file.name}"
    with open(temp_file, "wb") as file:
        file.write(loaded_file.getvalue())
        # file_name = loaded_file.name

    return temp_file

def get_loader_by_file_extension(temp_file):
    file_split = os.path.splitext(temp_file)
    file_name = file_split[0]
    file_extension = file_split[1]
    print('file_extension - ', file_extension)

    
    if file_extension == '.pdf':
        loader = PyPDFLoader(temp_file)
        logging.info('Loader Created for PDF file')
    
    elif file_extension == '.txt':
        loader = TextLoader(temp_file)

    elif file_extension == '.csv':
        loader = CSVLoader(temp_file)

    else :
        loader = UnstructuredFileLoader(temp_file)

    return loader
