#!/usr/bin/env python3 

import os
from flask import Flask
import config
from flask_wtf.csrf import CSRFProtect

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', config.SECRET_KEY)
app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024  # Limits uploads to 1000MB
app.config.update(
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax'
)
csrf = CSRFProtect(app)

UPLOAD_FOLDER = 'C:\\Users\\wolfe\\OneDrive\\Desktop\\pinecone-test\\upload-folder'
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', config.OPENAI_API_KEY)
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', config.PINECONE_API_KEY)
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV', config.PINECONE_API_ENV)
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

from routes import *

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
