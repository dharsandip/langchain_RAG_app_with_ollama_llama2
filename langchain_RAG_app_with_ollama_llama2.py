
import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.document_loaders import PyPDFLoader
from tempfile import NamedTemporaryFile

load_dotenv()
os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv("HUGGINGFACEHUB_API_TOKEN")


model_name = "hkunlp/instructor-large"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embeddings = HuggingFaceInstructEmbeddings(
    model_name=model_name, 
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)

memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=100)

model_local = Ollama(model="llama2")
llm = model_local

def process_input(bytes_data, query):
    
    with NamedTemporaryFile(delete=False) as tmp:  # open a named temporary file
        tmp.write(bytes_data)                      # write data from the uploaded file into it
        data = PyPDFLoader(tmp.name).load()
    os.remove(tmp.name)                            # remove temp file    
    
    # loader = PyPDFLoader(pdf_file)
    # data = loader.load()   
    data_chunks = text_splitter.split_documents(data)
    
    vectorstore = Chroma.from_documents(
        documents=data_chunks,
        collection_name="rag-chroma",
        embedding=embeddings,
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever(), memory=memory)

    result = conversation_chain({"question": query})
    answer = result["answer"]

    return answer


def main():

    st.title("Application for Medical Document Query")
    html_temp = """
    <div style="background-color:blue;padding:10px">
    <h2 style="color:white;text-align:center;">Application for Medical Document Query with Langchain, RAG and Open Source Llama2 model</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    pdf_file = st.file_uploader("Upload File", type=["pdf"])
    
    if ((pdf_file is not None) and ('pdf' in str(pdf_file))):
        bytes_data = pdf_file.read()
    
        query = st.text_input("Question")
    
        answer=""
        if st.button("Query Document"):
            answer = process_input(bytes_data, query)
            st.success('Answer: {}'.format(answer))

        
if __name__=='__main__':
    main()
       
        
        

