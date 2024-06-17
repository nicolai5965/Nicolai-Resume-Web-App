###------------------------------------------------------------------------------------------------------------###
#Importing libraries
import asyncio
import datetime
import getpass
import json
import os
import uuid
from pathlib import Path
from typing import AsyncGenerator, List, Sequence

import nest_asyncio
import openai
import streamlit as st
from google.colab import userdata
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers.pydantic import PydanticOutputParser
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate
)
from langchain.schema import AIMessage, HumanMessage
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langgraph.graph import END, MessageGraph
from PIL import Image
from pydantic import BaseModel, Field

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Generate a unique ID
unique_id = uuid.uuid4().hex[0:8]

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
