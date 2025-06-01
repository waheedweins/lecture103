# Importing necessary modules and classes
import streamlit as st
from langchain_google_genai import GoogleGenerativeAI , GoogleGenerativeAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
import os
from qdrant_client import QdrantClient

# loading Gemini api key
os.environ['GOOGLE_API_KEY'] = st.secrets["GOOGLE_API_KEY"]


# initilizing llm and embedding model
llm = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# loading document

file_path = "data/think_and_grow_rich.pdf"
loader = PyPDFLoader(file_path)
doc = loader.load()
document = doc[18:]

# splitting text

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
texts = text_splitter.split_documents(document)

# loading qdrant secrets
url = st.secrets["qdrant"]["url"]
api_key = st.secrets["qdrant"]["api_key"]

#deleting collection
client = QdrantClient(url=url, api_key=api_key)
client.delete_collection(collection_name="first_document")

#adding new collection
qdrant = QdrantVectorStore.from_documents(
    texts,
    embeddings,
    url=url,
    prefer_grpc=True,
    api_key=api_key,
    collection_name="first_document"
)

retriever = qdrant.as_retriever(search_kwargs={"k": 50})
qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)
chat_history = []



# stream lit
st.set_page_config(page_title = 'RAG chatbot', page_icon= "🤖")
st.title("Ask think and grow rich")
st.markdown("This chatbot uses Retrieval-Augmented Generation (RAG) to answer questions from the book *Think and Grow Rich*.")
query = st.text_input("📩 Ask a question:")
if query:
    with st.spinner("Searching for the best answer..."):
        result = qa_chain.invoke({"question": query, "chat_history": chat_history})
        st.success(result["answer"])
        chat_history.append((query, result["answer"]))


