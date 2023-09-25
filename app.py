# External imports
import os
import requests
import logging
import tempfile
import mimetypes
import docx2txt
from io import BytesIO
from PyPDF2 import PdfReader
from pytube import YouTube
from flask import Flask, render_template, request
from flask_limiter import Limiter
from werkzeug.utils import secure_filename


# Internal imports
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import whisper
import pinecone
import config

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limits uploads to 16MB

# Configuration and Logging
logging.basicConfig(level=logging.DEBUG)
app.logger.info('Info level log')
app.logger.warning('Warning level log')

UPLOAD_FOLDER = 'C:\Users\wolfe\OneDrive\Desktop\pinecone-test\upload-folder'
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', config.OPENAI_API_KEY)
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', config.PINECONE_API_KEY)
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV', config.PINECONE_API_ENV)

if not OPENAI_API_KEY:
    app.logger.error('OPENAI_API_KEY not set in environment variables!')
    raise ValueError('Required environment variable not set!')

if not PINECONE_API_KEY:
    app.logger.error('PINECONE_API_KEY not set in environment variables!')
    raise ValueError('Required environment variable not set!')

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

# Load external services
model = whisper.load_model("base")

# Request Limiting Configuration
def get_remote_address():
    return request.remote_addr

limiter = Limiter(app, key_func=get_remote_address)  # Adjust based on your needs

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def check_content(content):
    # Call a content moderation API
    response = requests.post('https://content-moderation-api.example.com/check', data={'text': content})
    if response.json().get('safe'):
        return True
    else:
        return False, response.json().get('reason')

def index_plain_text(content):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    texts = text_splitter.split_documents(content)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    index_name = "wolfe-test"
    docsearch = Pinecone.from_texts(texts, embeddings, index_name=index_name)
    return docsearch

def index_pdf(content):
    if isinstance(content, str):  # Plain text content
        return index_plain_text(content)
    else:  # Assuming it's a Document instance, future handling for other document types can go here.
        return index_plain_text(content)

def extract_text_from_pdf(pdf_path):
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text
 
def perform_query(index, query):
    docs = index.similarity_search(query)
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', config.OENAI_API_KEY)
    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    chain = load_qa_chain(llm, chain_type="stuff")
    response = chain.run(input_documents=docs, question=query)
    return response
    
@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)  # Make sure filename is secure
        file.save(os.path.join(UPLOAD_FOLDER, filename))
        return 'File uploaded successfully'
    else:
        return "Invalid file type."

@app.route('/search', methods=['POST'])
def search():
    app.logger.debug('Received search request.')
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
            app.logger.debug('Search completed successfully.')
            return render_template('search_results.html', query=query, results=results)
        return "Please provide both a document and a query."
    except Exception as e:
        app.logger.error(f"Error encountered during search: {str(e)}")
        return render_template('error.html', error_message=str(e))

@app.route('/index_from_link', methods=['POST'])
def index_from_link():
    try:
        link = request.form['pdf_link']
        query = request.form['query']

        if link and query:
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
                os.unlink(temp_file_path)
        return "Please provide both a link and a query."
    except Exception as e:
        return render_template('error.html', error_message=str(e))

@app.route('/video_transcribe', methods=['POST'])
def video_transcribe():
    try:
        video_link = request.form['video_link']
        query = request.form['query']
        temp_dir = tempfile.TemporaryDirectory()
        temp_audio_mp3_path = os.path.join(temp_dir.name, 'youtube_audio_copy.mp3')
        if video_link and query:
            try:
                yt = YouTube(video_link)
                audio_stream = yt.streams.filter(only_audio=True).first()
                audio_stream.download(output_path=temp_dir.name, filename='youtube_audio_copy.mp3')
                result = model.transcribe(temp_audio_mp3_path)
                transcribed_text = result["text"]
                index = index_pdf(transcribed_text)
                results = perform_query(index, query)
                return render_template('search_results.html', query=query, results=results)
            except Exception as download_error:
                return f"Error downloading the YouTube video: {str(download_error)}"
            finally:
                if os.path.exists(temp_audio_mp3_path):
                    os.remove(temp_audio_mp3_path)
    except Exception as e:
        return render_template('error.html', error_message=str(e))

@app.route('/static/<path:filename>')
def serve_static(filename):
    return app.send_static_file(filename)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(413)
def request_entity_too_large(error):
    return "File too large. Max allowed size is 16MB.", 413

@app.route('/')
def search_form():
    return render_template('search_form.html')

if __name__ == '__main__':
    app.run(debug=True)
