import os
from flask import Flask
import config

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024  # Limits uploads to 1000MB

UPLOAD_FOLDER = 'C:\\Users\\wolfe\\OneDrive\\Desktop\\pinecone-test\\upload-folder'
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', config.OPENAI_API_KEY)
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', config.PINECONE_API_KEY)
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV', config.PINECONE_API_ENV)
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

from routes import *

if __name__ == '__main__':
    app.run(debug=True)
