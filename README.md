# Document Search App

## Project Overview
The Document Search App is a Flask-based web application designed for efficient and accurate searching within various document formats, including PDF, TXT, and DOCX. It leverages advanced text processing and search algorithms to provide a powerful tool for content retrieval and data analysis.

## Key Packages and Open Source Projects Utilized
- **Flask Framework**: Backbone of the web application, offering a dynamic and responsive user interface, and seamless backend integration.
- **PyPDF2 & docx2txt**: Essential for extracting text from PDF and DOCX files, crucial for the app's content processing.
- **OpenAIEmbeddings & Pinecone**: Utilized for generating text embeddings and indexing, offering advanced search functionalities.
- **whisper (OpenAI)**: Employed for transcribing audio content from YouTube videos, expanding search capabilities to multimedia content.
- **Python Standard Libraries**: Includes `os`, `tempfile`, and `mimetypes` for file handling and system operations.

## Functionality and Workflow
1. **Content Extraction**: Extracts text from uploaded files or fetched documents.
2. **Embedding Generation**: Converts text into numerical representations for efficient searches.
3. **Indexing with Pinecone**: Allows for rapid and scalable searches.
4. **User Interface**: Facilitated by Flask for uploading documents and inputting queries.
5. **Search and Query Processing**: Performs similarity searches in response to user queries.
6. **Video Transcription**: Transcribes audio from videos, making the content searchable.

## Objectives and Benefits
Designed to simplify searching and extracting information from various document formats, this app is particularly beneficial for those working with large volumes of textual data. The integration of advanced AI and machine learning techniques ensures accuracy and contextual relevance in search results.

For more information, detailed documentation, and usage guidelines, visit the project's GitHub repository: [Document Search App](https://github.com/yourusername/document-search-app).

## License
This project is licensed under the MIT License.

