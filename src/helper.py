from langchain.llms import CTransformers
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from src.prompt_template import *
import pickle
import streamlit as st
import os


def save_model(model_path, model):
    # Save the FAISS index to a pickle file
    with open(model_path, "wb") as f:
        pickle.dump(model, f)


def load_model(model_path):
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
            return model
        
def get_streamlit_instance():
    return st
        