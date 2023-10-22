###------------------------------------------------------------------------------------------------------------###
#Importing libraries
from pathlib import Path
import os
import streamlit as st
from PIL import Image
import webbrowser
import numpy as np 
import pandas as pd
import math
from numba import njit , jit
from tqdm import trange, tqdm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from collections import Counter
from stqdm import stqdm
import matplotlib.pyplot as plt
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

st.sidebar.download_button(
    label = "Download Bachelor's project",
    data = PDFbyte_project,
    file_name = Bachelorproject_file,
    mime = "application/octet-stream",)
st.sidebar.write("---")
###------------------------------------------------------------------------------------------------------------###
## Instroduction 
st.header("Introduction:")
Introduction = """This project was my physics bachelor's project that I did with my classmate Simon. We got the idea from our supervisor Stefania. We could either do a convolutional neural network, which had been done multiple times before, or we could do a graph neural network which isn't as widespread as CNN. We chose the GNN because it sounded super interesting, especially since it isn't as widespread. The reason I'm making this page is that I have made some significant changes to the parts of the code. These changes makes the code run significantly faster. With some parts taking multipul hours down to a couple of minutes. 
If you want to read our completed report you can download it from the sidebar, or look at the chanllenge you self click the Kaggle link.
"""
st.write(Introduction)

st.write('\n')
st.write("---")
###------------------------------------------------------------------------------------------------------------###
