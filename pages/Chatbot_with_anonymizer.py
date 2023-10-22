###------------------------------------------------------------------------------------------------------------###
#Importing libraries
from pathlib import Path
import os
import streamlit as st
import spacy
from presidio_anonymizer import PresidioReversibleAnonymizer
from presidio_anonymizer.entities import OperatorConfig
from faker import Faker
import langdetect
import re
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
Introduction = """This project was my physics bachelor's project that I did with my classmate Simon. We got the idea from our supervisor Stefania. We could either do a convolutional neural network, which had been done multiple times before, or we could do a graph neural network which isn't as widespread as CNN. We chose the GNN because it sounded super interesting, especially since it isn't as widespread. The reason I'm making this page is that I have made some significant changes to the parts of the code. These changes makes the code run significantly faster. With some parts taking multipul hours down to a couple of minutes. 
If you want to read our completed report you can download it from the sidebar, or look at the chanllenge you self click the Kaggle link.
"""
st.write(Introduction)

st.write('\n')
st.write("---")
###------------------------------------------------------------------------------------------------------------###
## Anonymizer code: 

class DocumentAnonymizer:
    def __init__(self, use_faker=True):
        # Initializes the DocumentAnonymizer with the necessary configurations.
        nlp_config = {
            "nlp_engine_name": "spacy",
            "models": [
                {"lang_code": "en", "model_name": "en_core_web_md"},
                {"lang_code": "da", "model_name": "da_core_news_md"},
            ],
        }
        self.anonymizer = PresidioReversibleAnonymizer(
            add_default_faker_operators=use_faker,
            languages_config=nlp_config,
            faker_seed=42,
        )
        self.detected_language = None

    def register_custom_patterns(self, patterns_list):
        # Add custom patterns to the anonymizer.
        if not patterns_list:
            print("No custom patterns provided. Using default recognizers.")
            return

        for pattern_details in patterns_list:
            score = pattern_details.get('score', 1)
            custom_pattern = Pattern(
                name=pattern_details['name'],
                regex=pattern_details['regex'],
                score=score
            )
            custom_recognizer = PatternRecognizer(
                supported_entity=pattern_details['supported_entity'],
                patterns=[custom_pattern]
            )
            self.anonymizer.add_recognizer(custom_recognizer)

    def reset_mapping(self):
        # Resets the deanonymizer mapping.
        self.anonymizer.reset_deanonymizer_mapping()

    def detect_language(self, text):
        # Detects the language of the given text.
        self.detected_language = langdetect.detect(text)
        return self.detected_language

    def initialize_faker_operators(self, locale, custom_faker_operators):
        # Sets up custom operators for Faker based on the given locale.
        fake = Faker(locale)
        operators = {}

        for operator_details in custom_faker_operators:
            entity_type = operator_details["entity_type"]
            faker_function = getattr(fake, operator_details["faker_method"])
            if operator_details.get("digits"):
                operators[entity_type] = OperatorConfig(
                    "custom", 
                    {"lambda": lambda x, f=faker_function, d=operator_details["digits"]: str(f(digits=d))}
                )
            else:
                operators[entity_type] = OperatorConfig(
                    "custom", 
                    {"lambda": lambda x, f=faker_function: str(f())}
                )

        # Add the custom faker operators to the anonymizer
        self.anonymizer.add_operators(operators)

    def anonymize_text(self, document_content, detected_language):
        # Anonymizes the given document content based on the detected language.
        anonymized = self.anonymizer.anonymize(document_content, language=detected_language)
        return str(anonymized)

    def highlight_pii(self, string):
        # Highlights the given string with potential PII colored for visibility.
        re.sub(
            r"(<[^>]*>)", 
            lambda m: "\033[31m" + m.group(1) + "\033[0m", 
            string
        )

    def anonymize_document_content(self, document_content, custom_faker_operators=None):
        # Main method to anonymize the given document content.
        detected_language = self.detect_language(document_content)
        if custom_faker_operators:
            self.initialize_faker_operators(detected_language, custom_faker_operators)
        anonymized_content = self.anonymize_text(document_content, detected_language)
        self.highlight_pii(anonymized_content)
        return anonymized_content  # Return the anonymized content

    def display_mapping(self):
        # Prints the mapping between original and anonymized content.
        pprint.pprint(self.anonymizer.deanonymizer_mapping)

    def deanonymize_text(self, anonymized_content):
        # Deanonymizes the given content using the stored mapping.
        return self.anonymizer.deanonymize(anonymized_content)

###------------------------------------------------------------------------------------------------------------###

# Create an instance of DocumentAnonymizer
document_anonymizer = DocumentAnonymizer()

# 2. Document Input
st.title("Anonymized Chatbot Interface")
document = st.text_area("Paste your document content here:")

# 3. Display Document
if document:
    st.subheader("Original Document")
    st.write(document)
    
    st.subheader("Anonymized Document")
    # Using the DocumentAnonymizer to get the anonymized content
    anonymized_content = document_anonymizer.anonymize_document_content(document)
    st.write(anonymized_content)

# 4. Chat Interface
st.subheader("Chat with the bot")
question = st.text_input("Ask a question based on the document:")
if question:
    st.write(f"You asked (anonymized): {question}")  # Placeholder, will be replaced with actual anonymization
    st.write("Bot's answer (deanonymized): Answer will appear here.")  # Placeholder

# 5. Memory Viewer
st.subheader("Chatbot's Memory")
st.write("Memory content will appear here.")  # Placeholder

