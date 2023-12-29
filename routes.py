from flask import render_template, request, Flask
from werkzeug.utils import secure_filename
import pythoncom
import os
import tempfile
import requests
from io import BytesIO
from pytube import YouTube
from flask_limiter import Limiter
import whisper
import pinecone
import docx2txt
import mimetypes
from app import app, UPLOAD_FOLDER, PINECONE_API_KEY, PINECONE_API_ENV
from utilities import allowed_file, read_content, index_pdf, perform_query, extract_text_from_pdf, limit_tokens

# Initialize external services
model = whisper.load_model("base")
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

# Request Limiting Configuration
def get_remote_address():
    return request.remote_addr

limiter = Limiter(app=app, key_func=get_remote_address)  # Adjust based on your needs

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
            file_extension = os.path.splitext(secure_filename(pdf.filename))[1].lower()
            temp_path = os.path.join(tempfile.gettempdir(), secure_filename(pdf.filename))
            pdf.save(temp_path)

            if file_extension in ['.pdf', '.txt', '.docx']:
                pdf_content = read_content(temp_path)
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
                    index = index_pdf(text_chunk)
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
def request_entity_too_large():
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

@app.route('/favicon.ico')
def ignore_favicon():
    return '', 204

