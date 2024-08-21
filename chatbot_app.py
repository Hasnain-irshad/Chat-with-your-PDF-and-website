import streamlit as st
st.set_page_config(layout="wide")
import os
import base64
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
import torch
import requests
from bs4 import BeautifulSoup
import re

device = torch.device('cpu')

# Model and tokenizer loading
checkpoint = "MBZUAI/LaMini-T5-738M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,
    device_map='auto',
    offload_folder='offload' 
)

checkpoint_summarize = "t5-small"
tokenizer_summarize = AutoTokenizer.from_pretrained(checkpoint_summarize)
base_model_summarize = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_summarize, device_map='auto', torch_dtype=torch.float32)

def llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model=base_model,
        tokenizer=tokenizer,
        max_length=600,  # Moderate max length
        min_length=50,
        do_sample=True,
        temperature=0.25,  # Balanced temperature
        top_p=0.85,  # Controlled randomness
        repetition_penalty=1.2,  # Prevents repetition
        no_repeat_ngram_size=2  # Prevents repeating bigrams
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

def summarize_text(text):
    pipe_sum = pipeline(
        'summarization',
        model=base_model_summarize,
        tokenizer=tokenizer_summarize,
        max_length=800, 
        min_length=100
    )
    result = pipe_sum(text)
    return result[0]['summary_text']

def qa_llm(documents):
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    # Creating the vector store with FAISS
    db = FAISS.from_documents(documents, embeddings)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    return qa

def process_answer(instruction, documents, show_references=False):
    qa = qa_llm(documents)
    generated_text = qa(instruction)
    retrieved_docs = generated_text['source_documents']
    merged_text = " ".join([doc.page_content for doc in retrieved_docs])
    answer = generated_text['result']
    
    if show_references:
        references = "\n\n".join([f"**Reference {i+1}:** {doc.page_content[:500]}..." for i, doc in enumerate(retrieved_docs)])
        return answer + "\n\n" + references
    return answer

def file_preprocessing(file):
    loader = PDFMinerLoader(file)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    final_texts = ""
    for text in texts:
        final_texts = final_texts + text.page_content
    return final_texts

def display_conversation(history):
    for i in range(len(history["generated"])):
        st.markdown(f"<div style='text-align: right; font-size: 16px;'><p>👤 {history['past'][i]}</p></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align: left; font-size: 16px;'><p>🤖 {history['generated'][i]}</p></div>", unsafe_allow_html=True)

def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def data_ingestion():
    texts = []
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                loader = PDFMinerLoader(os.path.join(root, file))
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=500)
                texts.extend(text_splitter.split_documents(documents))
    return texts

class Document:
    def __init__(self, text):
        self.page_content = text
        self.metadata = {} 

def fetch_website_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Define a list of CSS classes or IDs to exclude
    exclude_classes = ['footer', 'sidebar', 'quick-links', 'calendar', 'advertisement']
    exclude_ids = ['footer', 'sidebar', 'quick-links', 'calendar', 'advertisement']
    
    # Find and remove elements with specific classes
    for class_name in exclude_classes:
        for element in soup.find_all(class_='{}'.format(class_name)):
            element.decompose()

    # Find and remove elements with specific IDs
    for id_name in exclude_ids:
        for element in soup.find_all(id='{}'.format(id_name)):
            element.decompose()

    # Collect and join the text from paragraphs
    paragraphs = soup.find_all('p')
    website_text = ' '.join([para.get_text(strip=True) for para in paragraphs if para.get_text(strip=True)])

    # Further cleanup to remove common unwanted patterns
    website_text = re.sub(r'(Calendar|Events|Latest News|Hostel|Transport|Copyright|Privacy Policy|Developed by).*', '', website_text, flags=re.IGNORECASE)
    website_text = re.sub(r'\s+', ' ', website_text).strip()  # Normalize whitespace

    return website_text

def website_data_ingestion(website_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=500)
    documents = [Document(website_text)]
    texts = text_splitter.split_documents(documents)
    return texts

def main():
    st.markdown("<h1 style='text-align:center; color: blue;'>Chat with Your PDF or Website🦜📄🌐</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center; color: grey;'>Built by Hasnain Irshad</h3>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center; color: red;'>Select Your Interaction Mode👇</h2>", unsafe_allow_html=True)

    mode = st.selectbox("Choose Interaction Mode:", ["Select an Option", "PDF", "Website"])
    
    if mode == "PDF":
        st.markdown("<h2 style='text-align:center; color: red;'>Upload Your PDF below👇</h2>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=["pdf"])
        if uploaded_file is not None:
            file_details = {
                "name": uploaded_file.name,
                "type": uploaded_file.type,
                "size": uploaded_file.size
            }
            file_path = "docs/" + uploaded_file.name
            with open(file_path, "wb") as temp_file:
                temp_file.write(uploaded_file.read())

            col1, col2 = st.columns([1.5, 2])
            with col1:
                st.markdown("<h4 style color:black;'>File details</h4>", unsafe_allow_html=True)
                st.json(file_details)
                st.markdown("<h2 style='text-align:center; color: grey;'>PDF Preview</h2>", unsafe_allow_html=True)
                displayPDF(file_path)

            with col2:
                st.markdown("<h2 style='text-align:center; color: grey;'>Options</h2>", unsafe_allow_html=True)
                option = st.selectbox("Choose an option:", ["Select an Option", "Summarize", "Ask a Question"])

                if option == "Summarize":
                    st.markdown("<h2 style='text-align:center; color: grey;'>Document Summary</h2>", unsafe_allow_html=True)
                    with st.spinner("Summarizing the document..."):
                        input_text = file_preprocessing(file_path)
                        summary = summarize_text(input_text)
                    st.write("📜", summary)

                elif option == "Ask a Question":
                    with st.spinner("Creating embeddings..."):
                        texts = data_ingestion()
                    st.success("Embeddings created successfully!")
                    st.markdown("<h2 style='text-align:center; color: grey;'>Chat Here</h2>", unsafe_allow_html=True)
                    
                    # Add a checkbox for showing/hiding references
                    show_references = st.checkbox("Show Document References", value=False)
                    
                    user_input = st.text_input("Enter your query:")
                    if user_input:
                        with st.spinner("Processing your query..."):
                            answer = process_answer(user_input, texts, show_references)
                        st.write("🤖", answer)

    elif mode == "Website":
        url = st.text_input("Enter Website URL:")
        if url:
            st.markdown("<h2 style='text-align:center; color: grey;'>Fetching Website Content</h2>", unsafe_allow_html=True)
            website_text = fetch_website_text(url)
            st.markdown("<h2 style='text-align:center; color: grey;'>Chat Here</h2>", unsafe_allow_html=True)
            
            # Add a checkbox for showing/hiding references
            show_references = st.checkbox("Show Document References", value=False)
            
            user_input = st.text_input("Enter your query:")
            if user_input:
                with st.spinner("Processing your query..."):
                    documents = website_data_ingestion(website_text)
                    answer = process_answer(user_input, documents, show_references)
                st.write("🤖", answer)

if __name__ == "__main__":
    main()