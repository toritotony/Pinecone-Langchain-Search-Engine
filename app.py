import os
from flask import Flask, render_template, request
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import pinecone
import requests
import openai
import tempfile
import mimetypes
import docx2txt 
from io import BytesIO
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pytube import YouTube
import speech_recognition as sr

app = Flask(__name__)

def index_pdf(content):
    if isinstance(content, str):  # Plain text content
        texts = [content]
    else:  # Assuming it's a Document instance
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
        texts = text_splitter.split_documents(content)

    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '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')
    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', 'c538f9c7-d734-4326-8e2f-a10d88a5cfd7')
    PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV', 'gcp-starter')

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
    index_name = "wolfe-test"
    docsearch = Pinecone.from_texts(texts, embeddings, index_name=index_name)

    return docsearch

def extract_text_from_pdf(pdf_path):
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def perform_query(index, query):
    docs = index.similarity_search(query)

    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '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')
    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    chain = load_qa_chain(llm, chain_type="stuff")

    response = chain.run(input_documents=docs, question=query)
    return response

@app.route('/search', methods=['POST'])
def search():
    try:
        pdf = request.files['pdf']
        query = request.form['query']
        if pdf and query:
            # Check the file extension to determine the type of document
            if pdf.filename.endswith('.pdf'):
                pdf_content = pdf.read()
                index = index_pdf(pdf_content)
            elif pdf.filename.endswith('.txt'):
                txt_content = pdf.read().decode('utf-8')
                index = index_pdf(txt_content)
            elif pdf.filename.endswith('.docx'):
                docx_content = pdf.read()
                txt_content = docx2txt.process(BytesIO(docx_content))
                index = index_pdf(txt_content)
            else:
                return "Unsupported file format."

            results = perform_query(index, query)
            return render_template('search_results.html', query=query, results=results)

        return "Please provide both a document and a query."
    except Exception as e:
        return render_template('error.html',error_message=str(e))


@app.route('/index_from_link', methods=['POST'])
def index_from_link():
    try:
        link = request.form['pdf_link']
        query = request.form['query']

        if link and query:
            # Create a temporary file to store the downloaded content
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file_path = temp_file.name

            try:
                response = requests.get(link)
                if response.status_code == 200:
                    temp_file.write(response.content)
                else:
                    return "Failed to fetch content from the provided link."

                mime_type, _ = mimetypes.guess_type(link)
                if mime_type == 'application/pdf':
                    content_text = extract_text_from_pdf(temp_file_path)
                elif mime_type == 'text/plain':
                    with open(temp_file_path, 'r', encoding='utf-8') as txt_file:
                        content_text = txt_file.read()
                elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                    docx_content = temp_file.read()
                    content_text = docx2txt.process(BytesIO(docx_content))
                else:
                    return "Unsupported link format."

                index = index_pdf(content_text)
                results = perform_query(index, query)
                return render_template('search_results.html', query=query, results=results)
            finally:
                temp_file.close()
                os.unlink(temp_file_path)  # Remove the temporary file

        return "Please provide both a link and a query."
    except Exception as e:
        return render_template('error.html', error_message=str(e))

@app.route('/video_transcribe', methods=['POST'])
def video_transcribe():
    try:
        video_link = request.form['video_link']
        query = request.form['query']

        if video_link and query:
            try:
                yt = YouTube(video_link)
                stream = yt.streams.filter(only_audio=True).first()
                
                temp_audio_path = 'C:\\Users\\wolfe\\OneDrive\\Desktop\\pinecone-test\\youtube_audio.mp4'
                
                stream.download(output_path=os.path.dirname(temp_audio_path), filename='youtube_audio_copy.mp4')
            except Exception as download_error:
                return f"Error downloading the YouTube video: {str(download_error)}"

            recognizer = sr.Recognizer()
            with sr.AudioFile(temp_audio_path) as source:
                audio = recognizer.record(source)

            try:
                transcribed_text = recognizer.recognize_google(audio)
            except sr.UnknownValueError:
                return "Transcription could not be performed."

            index = index_pdf(transcribed_text)  
            results = perform_query(index, query)  

            return render_template('search_results.html', query=query, results=results)
        else:
            return "Please provide both a video link and a query."
    except Exception as e:
        return render_template('error.html', error_message=str(e))


@app.route('/static/<path:filename>')
def serve_static(filename):
    return app.send_static_file(filename)

@app.route('/')
def search_form():
    return render_template('search_form.html')

if __name__ == '__main__':
    app.run(debug=True)
