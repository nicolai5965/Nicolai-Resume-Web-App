
###------------------------------------------------------------------------------------------------------------###
#Importing libraries
from pathlib import Path
import os
import streamlit as st
from PIL import Image
import webbrowser
os.environ['NUMEXPR_MAX_THREADS'] = '12'
###------------------------------------------------------------------------------------------------------------###
# --- PATH SETTINGS ---
resume_file = "pages/Nicolai's Resume.pdf"
profile_pic = "pages/Images/Nicolai_Image.jpg"

# --- GENERAL SETTINGS ---
PAGE_TITLE = "Digital CV | Nicolai Søderberg"
PAGE_ICON = ":wave:"
NAME = "Nicolai Søderberg"
DESCRIPTION = """
With a strong foundation in Quantum Physics from Copenhagen University, I've developed a profound ability to analyze and manipulate complex data structures, uncovering insights to solve real-world challenges. My journey in physics has evolved into a passion for data science and machine learning, where I apply my problem-solving skills to derive meaningful interpretations from data. 

At the core of my expertise lies a deep understanding of data processing and visualization, honed through roles in system integration and data consultancy. My recent projects involve advanced techniques in machine learning, including neural networks and graph neural networks, showcasing my ability to innovate and adapt to the evolving field of data science.

I am constantly exploring new realms in AI and machine learning, with a keen interest in deep learning, data engineering, and the transformative potential of these technologies. My goal is to leverage my skills in data science and machine learning to drive impactful solutions, whether it's through innovative project development or strategic consultancy.
"""

EMAIL = "nicolai@newf-dreams.dk"
Location = "Holbæk, Denmark"
Phone_number = "+4527576097"
SOCIAL_MEDIA = {
    "LinkedIn": "https://www.linkedin.com/in/nicolai-s%C3%B8derberg-907680238/",
    "GitHub": "https://github.com/nicolai5965",
}

###------------------------------------------------------------------------------------------------------------###
#Starting the streamlit web app page
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
###------------------------------------------------------------------------------------------------------------###
# --- LOAD CSS, PDF & PROFIL PIC ---
with open(resume_file, "rb") as pdf_file:
    PDFbyte = pdf_file.read()
profile_pic = Image.open(profile_pic)

###------------------------------------------------------------------------------------------------------------###

col1, col2, col3 = st.columns(3, gap="small")
with col1:
    st.image(profile_pic, width=230)

with col2:
    st.title(NAME)
    st.write("	:book:", DESCRIPTION)
    st.write("📫", EMAIL)
    st.write("	:house:", Location)
    st.write("	:telephone_receiver:", Phone_number)
    st.download_button(
        label=" 📑 Download Resume",
        data=PDFbyte,
        file_name=resume_file,
        mime="application/octet-stream",
    )
###------------------------------------------------------------------------------------------------------------###
# --- WORK HISTORY ---
st.write('\n')
st.write("---")
st.subheader("Work History")

st.write("🚧", "**System Integration Specialist | Intrum**")
st.write("Date: 01/2023 - 01/2023")
st.write(
    """
    - Collaborating with clients for onboarding
    - Developing integration solutions
    - Providing technical guidance
    """
)

st.write("🚧", "**Data consultant | Mobtimizers**")
st.write("Date: 08/2022 - 01/2023")
st.write(
    """
    - Data collection/extraction with API calls and manually
    - Cleaning/manipulating data before storage
    - Automated data storage system development
    """
)
###------------------------------------------------------------------------------------------------------------###
# --- SKILLS ---
st.write('\n')
st.write("---")
st.subheader("Hard Skills")
st.write(
    """
    - 👩‍💻 Programming: Python (Scikit-learn, Pandas, TensorFlow, Transformers / LLMs, etc), SQL
    - 📊 Data Visualization: Google Sheets, Google Studio, Matplotlib, Seaborn, Data Pipeline, Web Scraping
    - 📚 Modeling: MLP, Logistic regression, linear regression, decision trees, k-nearest neighbors, Langchain, Neural Network Architectures, Vector Databases, Machine Learning Algorithms
    - 🗄️ Databases: Google sheets, MySQL, Google Data Studio, SQL Server Management Studio, API Web App / Streamlit
    """
)

# --- PERSONAL PROJECTS ---
st.subheader("Personal Projects")
st.write(
    """
    - TrackML Particle Tracking Challenge (Kaggle, 2022)
    - Development of a Seq2Seq Transformer model
    """
)
###------------------------------------------------------------------------------------------------------------###
# --- EDUCATION  ---
st.write('\n')
st.write("---")
st.subheader("EDUCATION")
st.write(":mortar_board:", "**Bachelor in Quantum Physics | Copenhagen University Science (Niels Bohr Institute)**")
st.write("Date: 09/2019 - 08/2022")
st.write("Relevant Courses")
st.write(
    """
    - Modeling and Analysis of Data
    - Computer science for physicists 
    - Numerical Methods in Physics
    - Statistical physics
    - Mathematics for physicists 1, 2 and 3
    - Bachelor Thesis: "Analysing the use of Graph Neural Networks for particle track reconstruction"
    """
)
###------------------------------------------------------------------------------------------------------------###
## Links

LinkedIn_link = '[My LinkedIn](https://www.linkedin.com/in/nicolai-s%C3%B8derberg-907680238/)'
st.sidebar.markdown(LinkedIn_link, unsafe_allow_html=True)

GitHub_link = '[My GitHub repo](https://github.com/nicolai5965)'
st.sidebar.markdown(GitHub_link, unsafe_allow_html=True)

st.sidebar.download_button(
    label = "Download Resume 👈",
    data = PDFbyte,
    file_name = "Resume/CV.pdf",
    mime = "application/octet-stream",
)
###------------------------------------------------------------------------------------------------------------###
