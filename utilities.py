import os
import config
import docx2txt
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from flask_wtf.csrf import CSRFProtect
from app import app, OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_API_ENV, ALLOWED_EXTENSIONS
import pinecone

#test
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', config.SECRET_KEY)
csrf = CSRFProtect(app)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def limit_tokens(text, max_chars=10000): 
    if len(text) <= max_chars:
        return [text]

    texts = []
    while text:
        chunk = text[:max_chars]
        texts.append(chunk)
        text = text[max_chars:]
    
    return texts

def read_content(file_path):
    app.logger.debug("Reading content")
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == '.pdf':
        with open(file_path, "rb") as f:
            pdf_reader = PdfReader(f)
            text = ''.join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
        return text
    elif file_extension == '.txt':
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    elif file_extension == '.docx':
        return docx2txt.process(file_path)
    else:
        return None  # Unsupported file type

def index_pdf(content):
    app.logger.debug(f"index_pdf: Function called with content type: {type(content)}")
    
    if not isinstance(content, str):  
        content = read_content(content)

    if content is None:
        app.logger.error("Unsupported content type or unable to read content.")
        return None
    
    texts = limit_tokens(content)
    embeddings = OpenAIEmbeddings(chunk_size=200, openai_api_key=OPENAI_API_KEY)
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
    index_name = "test"

    if index_name not in pinecone.list_indexes():
        pinecone.create_index(name=index_name, metric="cosine", dimension=1536)
    for text_chunk in texts:
        try:
            docsearch = Pinecone.from_texts([text_chunk], embeddings, index_name=index_name)
            app.logger.debug("Chunk indexed successfully.")
        except Exception as e:
            app.logger.error(f"Error while indexing chunk: {e}")

    return docsearch

def extract_text_from_pdf(pdf_path):
    app.logger.debug(f"extract_text_from_pdf: Function called with path: {pdf_path}")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"No file found at {pdf_path}")
    pdf_reader = PdfReader(pdf_path)
    text = ''
    count = len(pdf_reader.pages)
    for i in range(count):
        page = pdf_reader.pages[i]
        text += page.extract_text()
    return text

def perform_query(index, query):
    try:
        index_name = "test"
        docs = index.similarity_search(query)
        llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106", request_timeout=1500, max_tokens=2048, temperature=0, openai_api_key=OPENAI_API_KEY)
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=query)
        pinecone.delete_index(delete_all=True, name=index_name)
    except Exception as e:
        app.logger.error(f"Error while performing query: {e}")
        return (f"Error: {e}")
    return response
