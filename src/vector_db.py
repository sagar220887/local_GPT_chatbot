from langchain.vectorstores import FAISS
import pandas as pd
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from constants import *
from main import st


def store_data_in_vectordb(documents, embeddings):
    current_knowledge_base = load_vectordb(VECTOR_DB_DIRECTORY, embeddings)
    print('current_vectordb - ', current_knowledge_base)

    new_knowledge_base =FAISS.from_documents(documents, embeddings)
    print('new_knowledge_base - ', new_knowledge_base)

    # current_knowledge_base.add_documents(embeddings)

    # Saving the new vector DB
    new_knowledge_base.save_local(VECTOR_DB_DIRECTORY)
    # vector_dataframe = convert_vectordb_to_df(new_knowledge_base)
    # st.write(vector_dataframe)
    return new_knowledge_base


def load_vectordb(stored_directory, embeddings):
    loaded_vector_db = FAISS.load_local(stored_directory, embeddings)
    return loaded_vector_db


## TODO : Need to visit code to add documents to existing vectorDB
def get_vector_store(text_chunks, embeddings):
    #Convert the Text Chunks into Embeddings and Create a FAISS Vector Store
    knowledge_base =FAISS.from_documents(text_chunks, embeddings)
    knowledge_base.add_documents(embeddings)
    print('INSIDE get_vector_store :: knowledge_base created')
    return knowledge_base 


def convert_vectordb_to_df(vectorDb):
    vector_dict = vectorDb.docstore._dict
    data_rows = []

    for k in vector_dict.keys():
        doc_name = vector_dict[k].metadata['source'].split('/')[-1]
        page_number = vector_dict[k].metadata['page'] + 1
        content =  vector_dict[k].page_content
        data_rows.append({"chunk_id": k, "document": doc_name, "page": page_number, "content":content})

    vector_df = pd.DataFrame(data_rows)
    print(vector_df)
    return vector_df


def save_vectordb_to_local(vectordb, vector_db_directory):
    vectordb.save_local(vector_db_directory)

def delete_document(vectordb, document):
    vector_df = convert_vectordb_to_df(vectordb)
    chunk_list = vector_df.loc[vector_df['document'] == document]['chunk_id'].tolist()
    vectordb.delete(chunk_list)

def refresh_model(vectordb, llm):
    retriever = vectordb.as_retriever()
    model = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retruever = retriever)
    return model


def get_similiar_docs(vector_db, query,k=1,score=False):
  if score:
    similar_docs = vector_db.similarity_search_with_score(query,k=k)
  else:
    similar_docs = vector_db.similarity_search(query,k=k)
  return similar_docs

