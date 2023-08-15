from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
import os

loader = PyPDFLoader("../data/field-guide-to-data-science.pdf")
data = loader.load()

print (f'You have {len(data)} document(s) in your data')
print (f'There are {len(data[30].page_content)} characters in your document')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
texts = text_splitter.split_documents(data)

print (f'Now you have {len(texts)} documents')
