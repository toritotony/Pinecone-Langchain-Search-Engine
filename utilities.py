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
from flask import request
from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', config.SECRET_KEY)
csrf = CSRFProtect(app)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def limit_tokens(text, max_chars=2000): 
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
        return None  

def index_pdf(content):
    app.logger.debug(f"index_pdf: Function called with content type: {type(content)}")

    if not isinstance(content, str):  
        content = read_content(content)

    if content is None:
        app.logger.error("Unsupported content type or unable to read content.")
        return None

    texts = limit_tokens(content)
    model_name = 'text-embedding-ada-002'
    embeddings = OpenAIEmbeddings(model=model_name, chunk_size=200, openai_api_key=OPENAI_API_KEY)

    for text in texts:
        query_result = embeddings.embed_query(text)

    app.logger.debug("QUERY RESULT")
    app.logger.debug(query_result)
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

    app.logger.debug("DOCSEARCH")
    app.logger.debug(docsearch)
    query = request.form['query']
    results = docsearch.similarity_search_with_score(query)
    app.logger.debug("Test Results")
    app.logger.debug(results)
    return docsearch

def perform_query(index, query):
    index_name = "test"
    try:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106", request_timeout=1500, max_tokens=2048, temperature=0, openai_api_key=OPENAI_API_KEY)
        prompt_template = """Use the following pieces of context to answer the question at the end. If you're not positive, provide a response based on the data given. If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        Helpful Answer:"""
        
        PROMPT = PromptTemplate(
                template=prompt_template, input_variables=["context", "question"]
            )

        system_template = """Use the following pieces of context to answer the user's question.
            If you're not positive, provide a response based on the data given. If you don't know the answer, just say that you don't know, don't try to make up an answer.
            ----------------
            {context}"""
        
        messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]

        CHAT_PROMPT = ChatPromptTemplate.from_messages(messages)


        PROMPT_SELECTOR = ConditionalPromptSelector(
            default_prompt=PROMPT, conditionals=[(is_chat_model, CHAT_PROMPT)]
        )
        docs = index.similarity_search(query)
        app.logger.debug("DOCS")
        app.logger.debug(docs)
        app.logger.debug("INDEX")
        app.logger.debug(index)
        app.logger.debug("QUERY")
        app.logger.debug(query)
        chain = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT_SELECTOR.get_prompt(llm))
        app.logger.debug("CHAIN")
        app.logger.debug(chain)
        response = chain.run(input_documents=docs, question=query)
        app.logger.debug("RESPONSE")
        app.logger.debug(response)
        #pinecone.delete_index(name=index_name)
    except Exception as e:
        app.logger.error(f"Error while performing query: {e}")
    return response
