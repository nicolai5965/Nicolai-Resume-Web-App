###------------------------------------------------------------------------------------------------------------###
#Importing libraries
from pathlib import Path
import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from operator import itemgetter
from langchain.chat_models.openai import ChatOpenAI
from langchain.schema.runnable import RunnableMap, RunnablePassthrough, RunnableLambda
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import Document
from langchain_experimental.data_anonymizer import PresidioReversibleAnonymizer
import spacy
import re
import pprint
import pandas as pd
import logging
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import openai
import json
from PIL import Image

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
## Instroduction 
st.header("Introduction:")
Introduction = """Welcome to my project showcasing an anonymized chatbot interface! This page was created as part of my journey into learning how to build chatbots and use large language models (LLMs). The goal was to explore techniques for building a chatbot that can have natural conversations while protecting personal information.

The chatbot uses state-of-the-art LLMs from Anthropic to have engaging dialogs. To anonymize sensitive information, it leverages Presidio for entity recognition and Faker for data generation. The interface allows custom entity patterns and operators to be defined.

After anonymization, original information can still be recovered using the reversible mapping. This project demonstrates how to balance utility and privacy when applying AI chatbots. The code is open source on GitHub to encourage experimentation.

I'm excited to present this proof of concept for an anonymizing chatbot. An anonymizing chatbot can serve as an extra safeguard when handling personal and sensitive data. 

Please try it out and let me know your thoughts! Feel free to view the full report for additional details on the implementation.
"""
st.write(Introduction)

st.write('\n')
st.write("---")
