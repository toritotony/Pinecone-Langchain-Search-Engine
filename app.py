# External imports
import os
import requests
import tempfile
import mimetypes
import docx2txt
import pythoncom
from io import BytesIO
from PyPDF2 import PdfReader
from pytube import YouTube
from flask import Flask, render_template, request
from flask_limiter import Limiter
from werkzeug.utils import secure_filename


# Internal imports
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
import whisper
import pinecone
import config

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024  # Limits uploads to 1000MB

UPLOAD_FOLDER = 'C:\\Users\\wolfe\\OneDrive\\Desktop\\pinecone-test\\upload-folder'
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', config.OPENAI_API_KEY)
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', config.PINECONE_API_KEY)
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV', config.PINECONE_API_ENV)
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

if not OPENAI_API_KEY:
    app.logger.error('OPENAI_API_KEY not set in environment variables!')
    raise ValueError('Required environment variable not set!')

if not PINECONE_API_KEY:
    app.logger.error('PINECONE_API_KEY not set in environment variables!')
    raise ValueError('Required environment variable not set!')

# initialize the pinecone service
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

# Load external services
model = whisper.load_model("base")

# Request Limiting Configuration
def get_remote_address():
    return request.remote_addr

limiter = Limiter(app=app, key_func=get_remote_address)  # Adjust based on your needs

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def limit_tokens(text, max_chars=10000): 
    """
    Splits the text into chunks with a maximum number of characters.
    """
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
        # Extract text from PDF
        with open(file_path, "rb") as f:
            pdf_reader = PdfReader(f)
            text = ''.join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
        return text
    elif file_extension == '.txt':
        # Read text from TXT file
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    elif file_extension == '.docx':
        # Extract text from DOCX file
        return docx2txt.process(file_path)
    else:
        return None  # Unsupported file type

def index_pdf(content):
    """
    Indexes the given content in Pinecone after splitting it into smaller chunks.
    """
    app.logger.debug(f"index_pdf: Function called with content type: {type(content)}")
    
    if not isinstance(content, str):  
        content = read_content(content)

    if content is None:
        app.logger.error("Unsupported content type or unable to read content.")
        return None
    
    # Splitting the content into smaller chunks
    texts = limit_tokens(content)
    embeddings = OpenAIEmbeddings(chunk_size=200, openai_api_key=OPENAI_API_KEY)
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
    index_name = "test"

    for text_chunk in texts:
        try:
            # Index each chunk as a separate vector
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
        docs = index.similarity_search(query)
        llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106", request_timeout=1500, max_tokens=2048, temperature=0, openai_api_key=OPENAI_API_KEY)
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=query)
    except Exception as e:
        app.logger.error(f"Error while performing query: {e}")
        return (f"Error: {e}")
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
            file_extension = os.path.splitext(secure_filename(pdf.filename))[1].lower()
            temp_path = os.path.join(tempfile.gettempdir(), secure_filename(pdf.filename))
            pdf.save(temp_path)

            if file_extension in ['.pdf', '.txt', '.docx']:
                pdf_content = read_content(temp_path)  # Ensure this returns a single string
            else:
                return "Unsupported file format."

            if pdf_content is None:
                return "Error processing the file."

            index = index_pdf(pdf_content)
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
    app.logger.debug("index_from_link: Function called")
    try:
        link = request.form['pdf_link']
        query = request.form['query']
        app.logger.debug(f"Received link: {link} and query: {query}")

        if link and query:
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file_path = temp_file.name
            try:
                response = requests.get(link)
                app.logger.debug(f"HTTP response status: {response.status_code}")
                if response.status_code == 200:
                    temp_file.write(response.content)
                else:
                    return "Failed to fetch content from the provided link."
                mime_type, _ = mimetypes.guess_type(link)
                app.logger.debug(f"Determined MIME type: {mime_type}")
                if mime_type == 'application/pdf':
                    content_text = extract_text_from_pdf(temp_file_path)
                    content_text = limit_tokens(content_text)
                    app.logger.debug(f"Extracted content type: {type(content_text)}")
                elif mime_type == 'text/plain':
                    with open(temp_file_path, 'r', encoding='utf-8') as txt_file:
                        content_text = txt_file.read()
                        content_text = limit_tokens(content_text)
                        app.logger.debug(f"Extracted content type: {type(content_text)}")
                elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                    docx_content = temp_file.read()
                    content_text = docx2txt.process(BytesIO(docx_content))
                    content_text = limit_tokens(content_text)
                    app.logger.debug(f"Extracted content type: {type(content_text)}")
                else:
                    return "Unsupported link format."
                all_results = []
                for text_chunk in content_text:
                    index = index_pdf(text_chunk)  # Process each chunk individually
                    app.logger.debug(f"Index object created: {index}")
                    results = perform_query(index, query)
                    all_results.extend(results)
                return render_template('search_results.html', query=query, results=results)
            finally:
                temp_file.close()
                os.unlink(temp_file_path)
        return "Please provide both a link and a query."
    except Exception as e:
        app.logger.error(f"index_from_link: Error encountered - {str(e)}")
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
                app.logger.debug("Audio downloaded successfully.")
                result = model.transcribe(temp_audio_mp3_path, fp16=False)
                transcribed_text = result["text"]
                transcribed_texts = limit_tokens(transcribed_text)
                for text in transcribed_texts:
                    index = index_pdf(text)
                    results = perform_query(index, query)
                app.logger.debug(transcribed_text)
                index = index_pdf(transcribed_text)
                app.logger.debug("Indexing successful")
                results = perform_query(index, query)
                app.logger.debug("Query successful")
                return render_template('search_results.html', query=query, results=results)
            except Exception as download_error:
                app.logger.error(f"Error downloading: {e}")
                return f"Error downloading the YouTube video: {str(download_error)}"
            finally:
                if os.path.exists(temp_audio_mp3_path):
                    os.remove(temp_audio_mp3_path)
    except Exception as e:
        app.logger.error(f"Error in video_transcribe: {e}")
        return render_template('error.html', error_message=str(e))

@app.route('/static/<path:filename>')
def serve_static(filename):
    return app.send_static_file(filename)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(413)
def request_entity_too_large(e):
    return "File too large. Max allowed size is 16MB.", 413

@app.route('/')
def search_form():
    return render_template('index.html')

@app.route('/document_search')
def document_search():
    return render_template('document_search.html')

@app.route('/link_search')
def link_search():
    return render_template('link_search.html')

@app.route('/video_search')
def video_search():
    return render_template('video_search.html')

if __name__ == '__main__':
    app.run(debug=True)
