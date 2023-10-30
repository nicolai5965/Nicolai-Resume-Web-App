###------------------------------------------------------------------------------------------------------------###
#Importing libraries
from pathlib import Path
import os
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models.openai import ChatOpenAI
from langchain.schema.runnable import RunnableMap, RunnablePassthrough, RunnableLambda
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import Document
from langchain_experimental.data_anonymizer import PresidioReversibleAnonymizer
import spacy
import re
import pprint
import langdetect
from presidio_analyzer import Pattern, PatternRecognizer
from presidio_anonymizer.entities import OperatorConfig
from faker import Faker
import pandas as pd
import logging
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import openai
from langchain.document_loaders import UnstructuredFileLoader

os.environ['NUMEXPR_MAX_THREADS'] = '12'
###------------------------------------------------------------------------------------------------------------###
# --- GENERAL SETTINGS ---
resume_file = "pages/Nicolai's Resume.pdf"
Bachelorproject_file = "pages/Bachelorproject.pdf"
SOCIAL_MEDIA = {
    "LinkedIn": "https://www.linkedin.com/in/nicolai-s%C3%B8derberg-907680238/",
    "GitHub": "https://github.com/nicolai5965",
}
###------------------------------------------------------------------------------------------------------------###
### Sidebar
st.sidebar.write('\n')

LinkedIn_link = '[My LinkedIn](https://www.linkedin.com/in/nicolai-s%C3%B8derberg-907680238/)'
st.sidebar.markdown(LinkedIn_link, unsafe_allow_html=True)

GitHub_link = '[My GitHub repo](https://github.com/nicolai5965)'
st.sidebar.markdown(GitHub_link, unsafe_allow_html=True)

with open(resume_file, "rb") as pdf_file:
    PDFbyte_CV = pdf_file.read()
    
st.sidebar.download_button(
    label = "Download Resume 👈",
    data = PDFbyte_CV,
    file_name = "Resume/CV.pdf",
    mime ="application/octet-stream",)

st.sidebar.write("---")

###------------------------------------------------------------------------------------------------------------###

def load_content(file):
    loader = UnstructuredFileLoader(file)
    docs = loader.load()
    return docs[0].page_content


st.title("!!! Still under process !!!")

st.title("Document Anonymizer")

# Text input
user_input = st.text_area("Paste your document here:")

# File uploader
uploaded_file = st.file_uploader("Or upload a file:", type=["txt", "pdf", "html", "docx"])

if uploaded_file:
    content = load_content(uploaded_file)
    st.write("Loaded content from file:")
    st.write(content)
elif user_input:
    st.write("You entered:")
    st.write(user_input)