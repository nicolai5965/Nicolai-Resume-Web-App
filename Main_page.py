
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
resume_file = "pages/CV.pdf"
profile_pic = "pages/Images/Nicolai_Image.jpg"

# --- GENERAL SETTINGS ---
PAGE_TITLE = "Digital CV | Nicolai Søderberg"
PAGE_ICON = ":wave:"
NAME = "Nicolai Søderberg"
DESCRIPTION = """
Data consultant with experience in data engineering, science, analytics and more.
"""
EMAIL = "nicolai@newf-dreams.dk"
Location = "Holbæk, Denmark"
Phone_number = "+4527576097"
SOCIAL_MEDIA = {
    "LinkedIn": "https://www.linkedin.com/in/nicolai-s%C3%B8derberg-907680238/",
    "GitHub": "https://github.com/nicolai5965",
}
PROJECTS = {
    "🏆 Sales Dashboard - Comparing sales across three stores": "https://youtu.be/Sb0A9i6d320",
    "🏆 Income and Expense Tracker - Web app with NoSQL database": "https://youtu.be/3egaMfE9388",
    "🏆 Desktop Application - Excel2CSV converter with user settings & menubar": "https://youtu.be/LzCfNanQ_9c",
    "🏆 MyToolBelt - Custom MS Excel add-in to combine Python & Excel": "https://pythonandvba.com/mytoolbelt/",
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
# --- HERO SECTION ---
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

st.write("🚧", "**Data consultant | Mobtimizers**")
st.write("Date:  08/2022 - Present")
st.write(
    """
- ► Data collection/extraction with API calls and manually.
- ► Cleaning/manipulating data before storaged manipulating 
- ► Saved the company multiply hours an week by making a automated data storaged system to allways have a clean and updated data storage
"""
)
###------------------------------------------------------------------------------------------------------------###
# --- SKILLS ---
st.write('\n')
st.write("---")
st.subheader("Hard Skills")
st.write(
    """
- 👩‍💻 Programming: Python (Scikit-learn, Pandas, etc), SQL, Julia (Novice)
- 📊 Data Visulization: Google Sheets, Google Studio, Matplotlib, Seaborn 
- 📚 Modeling: MLP , Logistic regression, linear regression, decition trees, k-nearest neighbors
- 🗄️ Databases: Goolge sheets, MySQL
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
- ► Modeling and Analysis of Data.
- ► Computer science for physicists 
- ► Numerical Methods in Physics
- ► Statistical physics
- ►Mathematics for physicists 1, 2 and 3
"""
)
###------------------------------------------------------------------------------------------------------------###
## Links

link = '[GitHub](http://github.com)'
st.markdown(f"""{link}<button style="background-color:Red;"></a>""", unsafe_allow_html=True)

st.markdown(f'''
<a href={link}><button style="background-color:Red;">Github<button/></a>
''', unsafe_allow_html=True)

st.markdown(f'''
<a href={link}><button style="background-color:Red;">Github<button/></a>
''', unsafe_allow_html=True)

st.markdown(f'''
<a{link}><button style="background-color:Red;">Github<button/></a>
''', unsafe_allow_html=True)

login = st.button(‘Login’)
if login:
webbrowser.open(‘http://github.com’)


if st.sidebar.button('LinkedIn'):
    webbrowser.open_new_tab(list(SOCIAL_MEDIA.values())[0])

if st.sidebar.button('GitHub'):
    #webbrowser.open_new_tab(link, unsafe_allow_html=True)
    st.markdown(link, unsafe_allow_html=True)
    
st.sidebar.download_button(
    label = "Download Resume 👈",
    data = PDFbyte,
    file_name = "Resume/CV.pdf",
    mime = "application/octet-stream",
)
###------------------------------------------------------------------------------------------------------------###
