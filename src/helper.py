from langchain.llms import CTransformers
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from src.prompt_template import *
import pickle


def create_huggingface_embeddings():
    #Load the Embedding Model
    embeddings=HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2', 
        model_kwargs={'device':'cpu'}
    )
    print('Inside create_huggingface_embeddings :: Embedding created')
    return embeddings




# def get_qa_prompt():
#     qa_prompt=PromptTemplate(template=get_qa_template(), input_variables=['context', 'question'])
#     return qa_prompt




def create_retrieval_chain(llm, vector_store, prompt):
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=vector_store.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=False,
        chain_type_kwargs={'prompt': prompt}
    )
    return chain


def create_conversational_chain(llm, vector_store, prompt):
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type='stuff',
        retriever=vector_store.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=False,
        memory=memory,
        chain_type_kwargs={'prompt': prompt}
    )
    return conversational_chain




def save_model(model_path, model):
    # Save the FAISS index to a pickle file
    with open(model_path, "wb") as f:
        pickle.dump(model, f)


def load_model(model_path):
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
            return model
        