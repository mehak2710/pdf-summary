import streamlit as st
import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text[:3000]  # Keeping it short for fast summarization

def summarize_with_groq(text):
    prompt = f"Summarize the following PDF content into exactly 15 lines:\n\n{text}"
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-8b-8192"
    )
    return response.choices[0].message.content.strip()

# Streamlit UI
st.title("📄 PDF to 15-Line Summary (Groq + Streamlit)")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Extracting and summarizing..."):
        extracted_text = extract_text_from_pdf(uploaded_file)
        summary = summarize_with_groq(extracted_text)
    st.subheader("📝 15-Line Summary")
    st.text(summary)
