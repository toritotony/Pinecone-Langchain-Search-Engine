# Document Search App

---

## Project Overview

The Document Search App is a Flask-based web application designed for efficient and accurate searching within various document formats, including PDF, TXT, and DOCX. Additionally, it enables querying of linked documents such as publicly available PDFs and transcribes YouTube videos to expand search capabilities to multimedia content. By leveraging advanced text processing, search algorithms, and embedding techniques, this app delivers a powerful solution for content retrieval and data analysis.

---

## Key Features

1. **Content Uploads**: Supports PDF, DOCX, and TXT file uploads for extracting and indexing text content.
2. **Link-Based Search**: Fetches and indexes documents available at publicly accessible links.
3. **Multimedia Support**: Transcribes audio from YouTube videos, making them searchable.
4. **Scalable Search**: Employs Pinecone for generating embeddings and indexing, enabling quick and accurate similarity searches.
5. **Interactive User Interface**: Built with Flask, featuring user-friendly forms and real-time feedback mechanisms.
6. **AI Integration**: Integrates OpenAI’s Whisper and embedding models for transcription and intelligent query responses.

---

## Technology Stack

- **Backend Framework**: Flask
- **Frontend**: HTML, CSS, JavaScript
- **Text Processing**: PyPDF2, docx2txt
- **Embedding and Indexing**: Langchain, Pinecone
- **Multimedia Handling**: OpenAI Whisper, Pytube
- **APIs**: OpenAI, Pinecone

---

## Workflow

1. **Document Handling**:
   - Uploads are processed to extract content using `PyPDF2`, `docx2txt`, or plain-text reading.
   - Publicly linked documents are fetched and processed similarly.

2. **Video Transcription**:
   - YouTube videos are downloaded and transcribed using OpenAI Whisper.

3. **Embedding and Indexing**:
   - Content is converted to embeddings using Langchain with OpenAI models.
   - Pinecone manages indexing for efficient search and retrieval.

4. **Query Processing**:
   - User-submitted queries are matched against indexed embeddings to provide relevant results.

5. **Result Display**:
   - The app presents results in a structured format via Flask templates.

--- 

## Local Deployment

This application can be deployed on any Flask-compatible environment. Production deployment can utilize Gunicorn or other WSGI servers for performance.

1. Clone the repository:
   ```
   git clone https://github.com/toritotony/Pinecone-Langchain-Search-Engine.git
   cd Pinecone-Langchain-Search-Engine
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv vnev 
   source venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up configuration:
   - Create a config.py file with key=value pairs 

5. Run the application:
   ```
   flask run
   ```

---

## Azure Deployment

For deploying the app on Azure App Service with GitHub Actions:

### Set Up Azure Resources:
- Create a resource group and an App Service in Azure.
- Configure the runtime stack, OS, region, and pricing plan.

### Connect to GitHub:
- Use Azure Deployment Center to link the GitHub repository.
- Enable continuous deployment for automatic updates on branch pushes.

### Create Startup Commands:
Set up commands to install dependencies and run the Gunicorn server:
```
gunicorn --bind 0.0.0.0 --timeout 600 --chdir /app app:app
```

### Configure GitHub Actions:
- Add Azure credentials to GitHub Actions secrets.
- Update the workflow file to:
  - Log into Azure CLI via a service principal.
  - Create a virtual environment, install dependencies, and deploy the app.

### Update Configuration:
- Ensure API keys and other secrets are configured in Azure settings.

### Test and Monitor:
- Push changes to trigger the workflow and test the deployed application.

For more detailed steps, refer to the following [Document](https://www.linkedin.com/feed/update/urn:li:activity:7191082240196419584/) or [LinkedIn Post](docs/Azure%20Web%20App%20Auto-Deployment%20via%20Github%20Actions.docx).

---

## Key Packages

- **Flask**: Provides the core web framework.
- **PyPDF2** and **docx2txt**: Extract text from PDF and DOCX files.
- **Pytube**: Downloads YouTube videos for transcription.
- **Whisper**: Converts audio content into text.
- **Pinecone**: Manages embeddings and indexes for similarity search.
- **Langchain**: Integrates with OpenAI’s embedding and LLM models.

---

## How to Use

1. Navigate to the application’s homepage.
2. Select the desired search method:
   - Upload a document.
   - Provide a public link to a document.
   - Enter a YouTube video link.
3. Input your query and execute the search.
4. View results in the formatted output page.

For more about the app functionality and features, refer to my [LinkedIn](https://www.linkedin.com/in/anthony-wolfe-102296237/).

---

## Objectives and Benefits

The app simplifies the process of searching and extracting information from various content types, including multimedia, providing:

- Time-efficient data retrieval.
- AI-powered, context-aware search results.
- Seamless support for diverse content types.

---

## Contribution

Contributions are welcome! Fork the repository and submit a pull request for new features, bug fixes, or improvements.

---

## License

This project is licensed under the MIT License. See this [Page](https://opensource.org/license/mit) for more details.
