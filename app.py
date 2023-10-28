# External imports
import os
import requests
import logging
import tempfile
import mimetypes
import docx2txt
import docx2pdf
import pythoncom
from docx2pdf import convert
from io import BytesIO
from PyPDF2 import PdfReader, PdfFileWriter, PdfFileReader, PageObject
from pytube import YouTube
from flask import Flask, render_template, request
from flask_limiter import Limiter
from werkzeug.utils import secure_filename
from docx import Document
from reportlab.pdfgen import canvas


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
app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024  # Limits uploads to 1000MB

# Configuration and Logging
logging.basicConfig(level=logging.DEBUG)
app.logger.info('Info level log')
app.logger.warning('Warning level log')

UPLOAD_FOLDER = 'C:\\Users\\wolfe\\OneDrive\\Desktop\\pinecone-test\\upload-folder'
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

limiter = Limiter(app=app, key_func=get_remote_address)  # Adjust based on your needs

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def limit_tokens(text, max_tokens=3900, buffer_tokens=260): 
    num_tokens = len(text.split())  
    if num_tokens <= max_tokens - buffer_tokens:
        return [text]

    words = text.split()
    texts = []
    truncated_text = ""

    while words:
        while words and len(truncated_text.split()) + len(words[0].split()) <= max_tokens - buffer_tokens:
            truncated_text += words.pop(0) + " "
        texts.append(truncated_text)
        truncated_text = ""
    
    return texts

def index_pdf(content):
    if isinstance(content, str):  
        texts = [content]
    else:  
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(content)

    texts = [text for text in texts if len(text) <= 3900]
    texts = [subtext for text in texts for subtext in limit_tokens(text)]

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
    index_name = "test"
    try:
        docsearch = Pinecone.from_texts(texts, embeddings, index_name=index_name)
    except Exception as e:
        print(f"Error while indexing: {e}")
        return None
    return docsearch

def extract_text_from_pdf(pdf_path):
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text
 
def perform_query(index, query):
    docs = index.similarity_search(query)
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
    pythoncom.CoInitialize()
    app.logger.debug('Received search request.')
    try:
        pdf = request.files['pdf']
        query = request.form['query']
        if pdf and query:
            # Check the file extension to determine the type of document
            if pdf.filename.endswith('.pdf'):
                # Save the uploaded file temporarily
                temp_path = os.path.join(tempfile.gettempdir(), secure_filename(pdf.filename))
                pdf.save(temp_path)
                # Extract text from the saved PDF
                pdf_content = extract_text_from_pdf(temp_path)
                pdf_content = limit_tokens(pdf_content)
                # Remove the temporary file
                os.remove(temp_path)
                index = index_pdf(pdf_content)
            elif pdf.filename.endswith('.txt'):
                temp_path = os.path.join(tempfile.gettempdir(), secure_filename(pdf.filename))
                pdf.save(temp_path)
                
                # Convert txt to pdf
                pdf_path = os.path.join(tempfile.gettempdir(), secure_filename(pdf.filename) + ".pdf")
                pdf_writer = PdfFileWriter()
                with open(temp_path, 'r', encoding='utf-8') as txt_file:
                    txt_content = txt_file.read()
                    pdf_page = pdf_writer.addPage(PageObject.createBlankPage(width=595, height=842))
                    pdf_page.drawText(txt_content)
                with open(pdf_path, "wb") as pdf_file:
                    pdf_writer.write(pdf_file)
                
                os.remove(temp_path)  # Remove the temporary txt file
                
                # Extract text from the converted PDF
                pdf_content = extract_text_from_pdf(pdf_path)
                pdf_content = limit_tokens(pdf_content)
                os.remove(pdf_path)  # Remove the temporary PDF file
                index = index_pdf(pdf_content)

            elif pdf.filename.endswith('.docx'):
                temp_path = os.path.join(tempfile.gettempdir(), secure_filename(pdf.filename))
                pdf.save(temp_path)
                
                # Convert docx to pdf
                pdf_path = temp_path + ".pdf"
                docx2pdf.convert(temp_path, pdf_path)
                os.remove(temp_path)  # Remove the temporary docx file
                
                # Extract text from the converted PDF
                pdf_content = extract_text_from_pdf(pdf_path)
                pdf_content = limit_tokens(pdf_content)
                os.remove(pdf_path)  # Remove the temporary PDF file
                index = index_pdf(pdf_content)
            else:
                return "Unsupported file format."
            results = perform_query(index, query)
            app.logger.debug('Search completed successfully.')
            return render_template('search_results.html', query=query, results=results)
        return "Please provide both a document and a query."
    except Exception as e:
        import traceback
        app.logger.error(f"Error encountered during search: {str(e)}")
        app.logger.error(traceback.format_exc())
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
                    content_text = limit_tokens(content_text)
                elif mime_type == 'text/plain':
                    with open(temp_file_path, 'r', encoding='utf-8') as txt_file:
                        content_text = txt_file.read()
                        content_text = limit_tokens(content_text)
                elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                    docx_content = temp_file.read()
                    content_text = docx2txt.process(BytesIO(docx_content))
                    content_text = limit_tokens(content_text)
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
                print("Audio downloaded successfully.")
                result = model.transcribe(temp_audio_mp3_path)
                transcribed_text = result["text"]
                transcribed_texts = limit_tokens(transcribed_text)
                for text in transcribed_texts:
                    index = index_pdf(text)
                    results = perform_query(index, query)
                print(transcribed_text)
                index = index_pdf(transcribed_text)
                print("Indexing successful")
                results = perform_query(index, query)
                print("Query successful")
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
