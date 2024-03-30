from langchain.vectorstores import FAISS
import pandas as pd
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from constants import *


def store_data_in_vectordb(documents, embeddings):
    current_vectordb = load_vectordb(VECTOR_DB_DIRECTORY, embeddings)
    print('current_vectordb - ', current_vectordb)

    new_knowledge_base =FAISS.from_documents(documents, embeddings)
    print('new_knowledge_base - ', new_knowledge_base)

    # Saving the new vector DB
    new_knowledge_base.save_local(VECTOR_DB_DIRECTORY)
    return new_knowledge_base

    ## TODO
    # Adding new data to existing vector DB
#     updated_knowledge_base = new_knowledge_base.merge_from(current_vectordb)
#     print('updated_knowledge_base - ', updated_knowledge_base)

    # Saving the new vector DB
#     updated_knowledge_base.save_local(vector_db_directory)
#     return updated_knowledge_base


## TODO : Need to visit code to add documents to existing vectorDB
def get_vector_store(text_chunks, embeddings):
    print('INSIDE get_vector_store')
    print('text_chunks - ', text_chunks)
    print('embeddings - ', embeddings)
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

    vector_df = pd.Dataframe(data_rows)
    return vector_df



def load_vectordb(stored_directory, embeddings):
    loaded_vector_db = FAISS.load_local(stored_directory, embeddings)
    return loaded_vector_db

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

