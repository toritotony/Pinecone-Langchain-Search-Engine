from flask import render_template, request
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from werkzeug.utils import secure_filename
import config
import os
import tempfile
import requests
from io import BytesIO
from pytube import YouTube
from flask_limiter import Limiter
import whisper
import pinecone
from flask_wtf.csrf import CSRFProtect
from app import app, UPLOAD_FOLDER, PINECONE_API_KEY, PINECONE_API_ENV
from utilities import allowed_file, read_content, index_pdf, perform_query, limit_tokens

app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', config.SECRET_KEY)
csrf = CSRFProtect(app)

# Initialize external services
model = whisper.load_model("base")
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

# Request Limiting Configuration
def get_remote_address():
    return request.remote_addr

limiter = Limiter(app=app, key_func=get_remote_address)  # Adjust based on your needs

class MyForm(FlaskForm):
    input_field = StringField('Input Field')
    submit_button = SubmitField('Submit')

@app.route('/upload', methods=['POST'])
def upload_file():
    form = MyForm()
    if form.validate_on_submit():
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)  # Make sure filename is secure
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            return 'File uploaded successfully'
        else:
            return "Invalid file type."

@app.route('/search', methods=['POST'])
def search():
    form = MyForm()
    if form.validate_on_submit():
        app.logger.debug('Received search request.')
        try:
            pdf = request.files['pdf']
            query = request.form['query']
            if pdf and query:
                file_extension = os.path.splitext(secure_filename(pdf.filename))[1].lower()
                temp_path = os.path.join(tempfile.gettempdir(), secure_filename(pdf.filename))
                pdf.save(temp_path)
                app.logger.debug("TEMPFILE PDF CONTENT:" + pdf.content_type)
                app.logger.debug("TEMPFILE PDF PATH:" + temp_path)

                if file_extension in ['.pdf', '.txt', '.docx']:
                    pdf_content = read_content(temp_path)
                    app.logger.debug("PDF CONTENT:" + pdf_content)
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
    form = MyForm()
    if form.validate_on_submit():
        app.logger.debug("index_from_link: Function called")
        try:
            link = request.form['pdf_link']
            query = request.form['query']
            app.logger.debug(f"Received link: {link} and query: {query}")

            if link and query:
                response = requests.get(link)
                app.logger.debug(f"HTTP response status: {response.status_code}")
                if response.status_code != 200:
                    return "Failed to fetch content from the provided link."
                
                # Write response content to a temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(link)[1])
                temp_file_path = temp_file.name
                temp_file.write(response.content)
                temp_file.close()
                app.logger.debug("TEMPFILE CONTENT:" + link)
                app.logger.debug("TEMPFILE PATH:" + temp_file_path)

                file_extension = os.path.splitext(link)[1].lower()
                if file_extension in ['.pdf', '.txt', '.docx']:
                    pdf_content = read_content(temp_file_path)
                    app.logger.debug("PDF CONTENT:" + pdf_content)
                else:
                    os.unlink(temp_file_path)  # Clean up temp file
                    return "Unsupported link format."

                if pdf_content is None:
                    os.unlink(temp_file_path)  # Clean up temp file
                    return "Error processing the file."

                index = index_pdf(pdf_content)
                results = perform_query(index, query)
                os.unlink(temp_file_path)  # Clean up temp file after processing
                return render_template('search_results.html', query=query, results=results)
            
            return "Please provide both a link and a query."
        except Exception as e:
            import traceback
            app.logger.error(f"index_from_link: Error encountered - {str(e)}")
            app.logger.error(traceback.format_exc())
            return render_template('error.html', error_message=str(e))
        
@app.route('/video_transcribe', methods=['POST'])
def video_transcribe():
    form = MyForm()
    if form.validate_on_submit():
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
                except Exception as download_error:
                    app.logger.error(f"Error downloading: {download_error}")
                    return render_template('error.html', error_message=str(download_error))

                try:
                    result = model.transcribe(temp_audio_mp3_path, fp16=False)
                    transcribed_text = result["text"]
                    transcribed_texts = limit_tokens(transcribed_text)
                    for text in transcribed_texts:
                        aggregate_text = " ".join(text)
                    app.logger.debug(transcribed_text)
                except Exception as read_error:
                    app.logger.error(f"Error reading: {read_error}")
                    return render_template('error.html', error_message=str(read_error))

                try:
                    index = index_pdf(aggregate_text)
                    app.logger.debug("Indexing successful")
                except Exception as index_error:
                    app.logger.error(f"Error indexing: {index_error}")
                    return render_template('error.html', error_message=str(index_error))

                try:
                    results = perform_query(index, query)
                    app.logger.debug("Query successful")
                    return render_template('search_results.html', query=query, results=results)

                except Exception as query_error:
                    app.logger.error(f"Error querying: {query_error}")
                    return render_template('error.html', error_message=str(query_error))

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

