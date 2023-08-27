import os
from flask import Flask, render_template, request
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from tqdm import tqdm
import pinecone
import requests
import openai

app = Flask(__name__)

def index_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)

    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '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')
    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', 'c538f9c7-d734-4326-8e2f-a10d88a5cfd7')
    PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV', 'gcp-starter')

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
    index_name = "wolfe-test"
    docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)

    return docsearch

def perform_query(index, query):
    docs = index.similarity_search(query)

    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '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')
    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    chain = load_qa_chain(llm, chain_type="stuff")

    response = chain.run(input_documents=docs, question=query)
    return response

@app.route('/pdf_search', methods=['POST'])
def pdf_search():
    pdf_link = request.form['pdf_link']
    query = request.form['query']

    if pdf_link and query:
        pdf_response = requests.get(pdf_link)
        if pdf_response.status_code == 200:
            pdf_path = 'downloaded_pdf.pdf'
            with open(pdf_path, 'wb') as pdf_file:
                pdf_file.write(pdf_response.content)

            index = index_pdf(pdf_path)
            results = perform_query(index, query)

            return render_template('search_results.html', query=query, results=results)
        else:
            return "Failed to download the PDF from the provided link."

    return "Please provide both a PDF link and a query."

@app.route('/video_transcribe', methods=['POST'])
def video_transcribe():
    video_link = request.form['video_link']
    query = request.form['query']

    if video_link and query:
        openai.api_key = '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'  # Replace with your actual OpenAI API key

        transcription_result = openai.SpeechApi.create(data={'audio_url': video_link})

        if 'text' in transcription_result:
            transcribed_text = transcription_result['text']

            # Index the transcribed text and perform queries
            index = index_pdf(transcribed_text)
            results = perform_query(index, query)

            return render_template('search_results.html', query=query, results=results)
        else:
            return "Failed to transcribe the provided video link."

    return "Please provide both a video link and a query."

@app.route('/static/<path:filename>')
def serve_static(filename):
    return app.send_static_file(filename)

@app.route('/')
def search_form():
    return render_template('search_form.html')

@app.route('/search', methods=['POST'])
def search():
    pdf = request.files['pdf']
    query = request.form['query']

    if pdf and query:
        pdf_path = os.path.join('c:\\Users\\wolfe\\OneDrive\\Desktop\\BOOKS', pdf.filename)
        pdf.save(pdf_path)

        index = index_pdf(pdf_path)
        results = perform_query(index, query)

        return render_template('search_results.html', query=query, results=results)

    return "Please provide both a PDF and a query."

if __name__ == '__main__':
    app.run(debug=True)