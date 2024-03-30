from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory




def create_chain(llm, vector_store, prompt):
    chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=vector_store.as_retriever(search_kwargs={'k': 2}),
            return_source_documents=False,
            chain_type_kwargs={'prompt': prompt}
    )
    print('Chain created')
    return chain

def create_conversational_chain(llm, vector_store, prompt):
    memory = ConversationBufferMemory()
    conversation_chain = ConversationalRetrievalChain(
        llm=llm,
        chain_type='stuff',
        retriever=vector_store.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt},
        memory=memory
    )
    print('Chain created')
    return conversation_chain