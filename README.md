# Chat with Your PDF or Website 🦜📄🌐

## Overview

This Streamlit application allows users to interact with PDF documents and websites by summarizing their contents or answering questions based on the text. The application utilizes the `transformers` library for model handling and `langchain` for document processing and retrieval.

## Features

### PDF Interaction:
- Upload PDF files and get them summarized.
- Ask questions related to the content of the uploaded PDF.

### Website Interaction:
- Fetch content from any public website.
- Summarize the website content.
- Ask questions based on the fetched website content.

## Installation

To run this application, you'll need to have Python 3.7 or later installed. You can install the required dependencies using `pip`. Follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/chat-with-pdf-website.git
    cd chat-with-pdf-website
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Ensure you have a compatible environment for GPU acceleration if you plan to use it. Otherwise, the application will run on CPU.

## Usage

1. Start the Streamlit application:
    ```bash
    streamlit run app.py
    ```

2. Open your web browser and go to [http://localhost:8501](http://localhost:8501) to access the application.

## Code Explanation

### Model and Tokenizer Loading:
Models for text generation and summarization are loaded from the `transformers` library.

### PDF Processing:
PDFs are loaded and split into chunks using `PDFMinerLoader` and `RecursiveCharacterTextSplitter`.

### Website Processing:
Website content is fetched and cleaned using `BeautifulSoup` and regular expressions.

### Chat Interface:
Users can interact with the application by uploading PDFs or entering website URLs. They can then summarize content or ask questions based on the text.

## Configuration

### Models:
- `MBZUAI/LaMini-T5-738M` for question-answering.
- `t5-small` for summarization.

### Text Splitting:
- For PDFs: Chunk size of 200 characters with 50 characters overlap.
- For Websites: Chunk size of 500 characters with 500 characters overlap.

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request. All contributions are welcome!


## Contact

For any questions or suggestions, feel free to open an issue or contact [irshadhasnain827@gmail.com].
