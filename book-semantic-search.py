from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from tqdm.autonotebook import tqdm
import pinecone
import os

loader = PyPDFLoader("../data/field-guide-to-data-science.pdf")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
texts = text_splitter.split_documents(data)

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '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')

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', '9d837134-08fe-4023-a178-f361ae686679')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV', 'gcp-starter')

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
)
index_name = "wolfe-test"

docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)

query = "What's the easiest way to integrate?"
docs = docsearch.similarity_search(query)

llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
chain = load_qa_chain(llm, chain_type="stuff")

chain.run(input_documents=docs, question=query)

