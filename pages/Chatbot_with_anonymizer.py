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
import langdetect
from presidio_analyzer import Pattern, PatternRecognizer
from presidio_anonymizer.entities import OperatorConfig
from faker import Faker
import pandas as pd
import logging
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import openai


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
        try:
            self.detected_language = langdetect.detect(text)
        except langdetect.lang_detect_exception.LangDetectException:
            self.detected_language = "en"  # default to English
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
        return re.sub(r"(<[^>]*>)", lambda m: "**" + m.group(1) + "**", string
        )

    def anonymize_document_content(self, document_content, custom_faker_operators=None):
        # Main method to anonymize the given document content.
        detected_language = self.detect_language(document_content)
        if custom_faker_operators:
            self.initialize_faker_operators(detected_language, custom_faker_operators)

        # Check if the detected language is supported
        if detected_language not in ['en', 'da']:
            detected_language = 'en'  # Default to English if the language is not supported
        
        anonymized_content = self.anonymize_text(document_content, detected_language)
        self.highlight_pii(anonymized_content)
        return anonymized_content  # Return the anonymized content

    def display_mapping(self):
        # Prints the mapping between original and anonymized content.
        return pprint.pformat(self.anonymizer.deanonymizer_mapping)

    def deanonymize_text(self, anonymized_content):
        # Deanonymizes the given content using the stored mapping.
        return self.anonymizer.deanonymize(anonymized_content)

###------------------------------------------------------------------------------------------------------------###
# Chatbot code: 
class ChatbotMemory:
    # A class to represent a Chatbot that integrates memory and document anonymization capabilities.
    def __init__(self, document_anonymizer, document_content, openai_key):
        # Initializes the Chatbot with given document anonymizer, content, and OpenAI key.
        self.document_anonymizer = document_anonymizer
        self.document_content = document_content
        self.openai_key = openai_key
        self.setup()

    def setup(self):
        # Sets up the Chatbot components.
        self.documents = self.convert_to_document(self.document_content)
        self.anonymize_document()
        self.setup_text_splitter()
        self.setup_embeddings()
        self.setup_retriever()
        self.setup_prompt_and_model()
        self.setup_anonymizer_chain()

    def convert_to_document(self, document_content):
        # Converts the document content to a Document object.
        return [Document(page_content=document_content)]

    def anonymize_document(self):
        # Anonymizes the content of each document.
        for doc in self.documents:
            doc.page_content = self.document_anonymizer.anonymize_document_content(doc.page_content)

    def setup_text_splitter(self):
        # Initializes the text splitter.
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        self.chunks = self.text_splitter.split_documents(self.documents)

    def setup_embeddings(self):
        # Initializes the embeddings for OpenAI.
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_key)

    def setup_retriever(self):
        # Initializes the document retriever.
        docsearch = FAISS.from_documents(self.chunks, self.embeddings)
        self.retriever = docsearch.as_retriever()

    def setup_prompt_and_model(self):
        # Sets up the prompt and model for OpenAI.
        template = (
            """You are a chatbot having a conversation with a human.
            Given the following extracted parts of a long document and a question, create a final answer.
            {context}
            {chat_history}
            Human: {human_input}
            Chatbot:"""
        )
        self.prompt = PromptTemplate(
            input_variables=["chat_history", "human_input", "context"], template=template
        )
        self.model = ChatOpenAI(openai_api_key=self.openai_key, temperature=0)
        self.memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")

    def setup_anonymizer_chain(self):
        # Sets up the anonymizer chain for the chatbot.
        _inputs = RunnableMap(
            question=RunnablePassthrough(),
            anonymized_question=RunnableLambda(self.document_anonymizer.anonymize_document_content),
            chat_history=RunnablePassthrough()
        )
        self.anonymizer_chain = (
            _inputs
            | {
                "context": itemgetter("anonymized_question") | self.retriever,
                "human_input": itemgetter("anonymized_question"),
                "chat_history": lambda _: self.memory.load_memory_variables({})["chat_history"]
            }
            | self.prompt
            | self.model
            | StrOutputParser()
        )
        self.chain_with_deanonymization = self.anonymizer_chain | RunnableLambda(self.document_anonymizer.deanonymize_text)

    def view_document_before_anonymization(self):
        # Returns the original document before anonymization.
        return self.document_content

    def view_document_after_anonymization(self):
        # Returns the document after it has been anonymized.
        return [doc.page_content for doc in self.documents]

    def view_anonymized_question(self, question):
        # Returns the anonymized version of a given question.
        return self.document_anonymizer.anonymize_document_content(question)

    def view_answer(self, question):
        # Returns the answer to a given question, after processing through the chatbot.
        anonymized_question = self.view_anonymized_question(question)
        response = self.anonymizer_chain.invoke(anonymized_question)

        # Update memory with anonymized question and response.
        self.memory.save_context({"human_input": anonymized_question}, {"output": response})
        deanonymized_response = self.document_anonymizer.deanonymize_text(response)
        return deanonymized_response

    def get_memory_content(self):
        # Returns the content stored in the chatbot's memory.
        return self.memory.load_memory_variables({})

###------------------------------------------------------------------------------------------------------------###

# Define default patterns and operators
DEFAULT_FAKER_OPERATORS = [
    #{"entity_type": "PERSON", "faker_method": "name"},  # Modified this line
    {"entity_type": "LOCATION", "faker_method": "city"},
    {"entity_type": "SSN", "faker_method": "ssn"},
    {"entity_type": "FULL_US_DRIVER_LICENSE", "faker_method": "license_plate"},
    {"entity_type": "CPR", "faker_method": "random_number", "digits": 10},
    {"entity_type": "BANK_ACCOUNT", "faker_method": "random_number", "digits": 14},
    {"entity_type": "DANISH_PASSPORT", "faker_method": "random_number", "digits": 9},
    {"entity_type": "FULL_NAME", "faker_method": "name"},
    {"entity_type": "FIRST_NAME", "faker_method": "first_name"},
    {"entity_type": "LAST_NAME", "faker_method": "last_name"},
]

DEFAULT_PATTERNS = [
    {"name": "ssn_pattern", "regex": r"\b\d{3}-?\d{2}-?\d{4}\b", "supported_entity": "SSN", "score": 1},
    {"name": "cpr_pattern", "regex": r"\b\d{2}\d{2}\d{2}-?\d{4}\b", "supported_entity": "CPR", "score": 1},
    {"name": "danish_phone_pattern", "regex": r"\b\+45 ?\d{2} ?\d{2} ?\d{2} ?\d{2}\b", "supported_entity": "DANISH_PHONE", "score": 1},
    {"name": "credit_card_pattern", "regex": r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b", "supported_entity": "CREDIT_CARD", "score": 1},
    {"name": "bank_account_pattern", "regex": r"\b\d{4}[ -]?\d{10}\b", "supported_entity": "BANK_ACCOUNT", "score": 1},
    {"name": "drivers_license_pattern", "regex": r"\bD\d{3}-?\d{4}-?\d{4}-?\d\b", "supported_entity": "FULL_US_DRIVER_LICENSE", "score": 1},
    {"name": "danish_license_pattern", "regex": r"\b(DK)?\d{10}\b", "supported_entity": "DANISH_LICENSE", "score": 1},
    {"name": "danish_passport_pattern", "regex": r"\b(DK)?\d{9}\b", "supported_entity": "DANISH_PASSPORT", "score": 1},
    {"name": "full_name_pattern", "regex": r"(?P<FULL_NAME>[A-Z][a-z]+ [A-Z][a-z]+)", "supported_entity": "FULL_NAME", "score": 3},
    {"name": "first_name_pattern", "regex": r"\b(?P<FIRST_NAME>[A-Z][a-z]+)\b", "supported_entity": "FIRST_NAME", "score": 1},
    {"name": "last_name_pattern", "regex": r"\b(?P<LAST_NAME>[A-Z][a-z]+)\b", "supported_entity": "LAST_NAME", "score": 1},
    {"name": "date_pattern", "regex": r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}\b", "supported_entity": "DATE_TIME", "score": 1},
]

# Initialize custom patterns and operators with default values
custom_patterns = DEFAULT_PATTERNS.copy()
custom_faker_operators = DEFAULT_FAKER_OPERATORS.copy()

# Check if custom patterns and operators exist in session state
if 'custom_patterns' not in st.session_state:
    st.session_state.custom_patterns = DEFAULT_PATTERNS.copy()

if 'custom_faker_operators' not in st.session_state:
    st.session_state.custom_faker_operators = DEFAULT_FAKER_OPERATORS.copy()


# A dictionary to map language codes to their full names
language_name_mapping = {
    "en": "English",
    "da": "Danish"
    # Add more mappings as needed
}

# Initialize document with a default value
document = """Name: Nicolai Søderberg
Email: nicolai@newf-dreams.com
Phone: +4527576097
Address: 123 Main St, Copenhagen
Objective: Experienced software developer with a passion for creating innovative solutions. Seeking a challenging role in a forward-thinking tech company.
Skills: Python, Streamlit, Data Analysis, Machine Learning
Education: Bachelor's in Computer Science, University of Copenhagen, 2015-2018
Experience: System specialist, 2023->
"""
###------------------------------------------------------------------------------------------------------------###

st.title("Anonymized Chatbot Interface")

# 1. Anonymization Settings
use_faker = st.sidebar.checkbox("Use Faker", value=True)

# Toggle for using custom patterns in the sidebar
use_custom_patterns = st.sidebar.checkbox("Use Custom Patterns", value=True)
if use_custom_patterns != st.session_state.get('prev_use_custom_patterns', True):
    st.session_state.show_anonymizing = False  # Reset the anonymizer state
    st.session_state.start_chatbot = False     # Reset the chatbot state
    st.session_state.messages = []             # Clear the chat messages
    st.session_state.prev_use_custom_patterns = use_custom_patterns

# Toggle for using custom faker operators in the sidebar
use_custom_faker_operators = st.sidebar.checkbox("Use Custom Faker Operators", value=True)
if use_custom_faker_operators != st.session_state.get('prev_use_custom_faker_operators', True):
    st.session_state.show_anonymizing = False  # Reset the anonymizer state
    st.session_state.start_chatbot = False     # Reset the chatbot state
    st.session_state.messages = []             # Clear the chat messages
    st.session_state.prev_use_custom_faker_operators = use_custom_faker_operators


# 2. Document Input
document = st.text_area("Paste your document content here:", key="document_input", value=document)


# 3. Language Detection
#st.sidebar.header("Language Detection")
#detected_language = document_anonymizer.detect_language(document) if document else None
#language = st.sidebar.selectbox("Detected/Choose Language", ["Auto-detect", "English", "Danish"], index=0 if not detected_language else ["en", "da"].index(detected_language))
#if language != "Auto-detect":
#    detected_language = language

# 4. Custom Pattern Registration
with st.sidebar.expander("Custom Pattern Registration"):
    custom_pattern_name = st.text_input("Pattern Name")
    custom_pattern_regex = st.text_input("Regex Pattern")
    custom_pattern_entity = st.text_input("Supported Entity")

    add_pattern = st.button("Add Pattern")
    if add_pattern and custom_pattern_name and custom_pattern_regex and custom_pattern_entity:
        custom_pattern = {
            'name': custom_pattern_name,
            'regex': custom_pattern_regex,
            'supported_entity': custom_pattern_entity
        }
        st.session_state.custom_patterns.append(custom_pattern)
        document_anonymizer.register_custom_patterns([custom_pattern])

    reset_patterns = st.button("Reset to Default Patterns")
    if reset_patterns:
        st.session_state.custom_patterns = DEFAULT_PATTERNS.copy()
        document_anonymizer.register_custom_patterns(st.session_state.custom_patterns)

    if st.checkbox("View Added Patterns"):
        for pattern in st.session_state.custom_patterns:
            st.write(pattern)

# 4.1 Custom Faker Operator Registration
with st.sidebar.expander("Custom Faker Operator Registration"):
    entity_type = st.text_input("Entity Type (for Faker)")
    faker_method = st.text_input("Faker Method")
    digits = st.number_input("Digits (if applicable)", min_value=0, value=0, format="%d")

    add_operator = st.button("Add Faker Operator")
    if add_operator and entity_type and faker_method:
        custom_operator = {}
        if digits:
            custom_operator = {
                "entity_type": entity_type,
                "faker_method": faker_method,
                "digits": digits
            }
        else:
            custom_operator = {
                "entity_type": entity_type,
                "faker_method": faker_method
            }
        st.session_state.custom_faker_operators.append(custom_operator)
        detected_language = document_anonymizer.detect_language(document) if document else None
        document_anonymizer.initialize_faker_operators(detected_language, [custom_operator])

    if st.session_state.custom_faker_operators:
        reset_operators = st.button("Reset to Default Faker Operators")
        if reset_operators:
            st.session_state.custom_faker_operators = DEFAULT_FAKER_OPERATORS.copy()
            detected_language = document_anonymizer.detect_language(document) if document else None
            document_anonymizer.initialize_faker_operators(detected_language, st.session_state.custom_faker_operators)

        if st.checkbox("View Added Faker Operators"):
            for operator in st.session_state.custom_faker_operators:
                st.write(operator)

# Check if the 'show_anonymizing' state exists, if not, initialize it to False
if 'show_anonymizing' not in st.session_state:
    st.session_state.show_anonymizing = False

# The "Start Anonymizing" button should always be visible
start_anonymizing = st.button("Start Anonymizing")

# If the "Start Anonymizing" button is clicked, set 'show_anonymizing' to True
if start_anonymizing:
    st.session_state.show_anonymizing = True

# Only display the anonymizing section if 'show_anonymizing' is True
if st.session_state.show_anonymizing and document:
    # Initialize anonymizing
    document_anonymizer = DocumentAnonymizer(use_faker=use_faker)
    highlight_anonymizer = None  # Initialize to None

    detected_language = document_anonymizer.detect_language(document)

    # Ensure custom patterns from session state are registered
    if use_custom_patterns and 'custom_patterns' in st.session_state:
        document_anonymizer.register_custom_patterns(st.session_state.custom_patterns)

    # Ensure custom faker operators from session state are initialized
    if use_custom_faker_operators and 'custom_faker_operators' in st.session_state:
        document_anonymizer.initialize_faker_operators(detected_language, st.session_state.custom_faker_operators)

    # Display Detected Language
    with st.expander("Detected Language"):
        language_name = language_name_mapping.get(detected_language, "Unknown")
        st.write(f"The detected language is: {language_name} ({detected_language})")

    # 5. Display Document
    with st.expander("Original Document"):
        st.write(document)

    with st.expander("Anonymized Document"):
        # Using the DocumentAnonymizer to get the anonymized content
        anonymized_content = document_anonymizer.anonymize_document_content(document)
        st.write(anonymized_content)

    # 6. Mapping Viewer
    with st.expander("View Mapping"):
        mapping = document_anonymizer.display_mapping()
        st.write(mapping)

    # 7. Deanonymization Feature
    with st.expander("Deanonymize Content"):
        deanonymized_content = document_anonymizer.deanonymize_text(anonymized_content)
        st.subheader("Deanonymized Document")
        st.write(deanonymized_content)

    # 8. Highlighting PII
    with st.expander("Highlighted PII in Document"):
        if use_faker:  # If the main anonymizer uses faker, then create a separate highlight_anonymizer without faker
            highlight_anonymizer = DocumentAnonymizer(use_faker=False)
            highlight_anonymizer.register_custom_patterns(custom_patterns)
            highlight_anonymized_content = highlight_anonymizer.anonymize_document_content(document)
            highlighted_content = highlight_anonymizer.highlight_pii(highlight_anonymized_content)
        else:  # Else, just use the main anonymizer for highlighting
            highlighted_content = document_anonymizer.highlight_pii(anonymized_content)
        st.write(highlighted_content)

    # Add a close button at the end
    if st.button("Close Anonymizing"):
        st.session_state.show_anonymizing = False
        st.rerun()  # Force Streamlit to rerun the script immediately
 

###------------------------------------------------------------------------------------------------------------###

# Chatbot interface: 

st.title("💬 Chatbot")


openai_api_key = os.environ.get('OPENAI_API_KEY', None)
# Initialize reset_counter in session state if it doesn't exist
if 'reset_counter' not in st.session_state:
    st.session_state.reset_counter = 0

@st.cache_resource()
def initialize_chatbot(document_content, openai_api_key, reset_counter):
    # Initialize the DocumentAnonymizer and ChatbotMemory classes
    document_anonymizer_memory = DocumentAnonymizer(use_faker=True)
    detected_language = document_anonymizer_memory.detect_language(document_content)
    # Ensure custom patterns from session state are registered
    if use_custom_patterns and 'custom_patterns' in st.session_state:
        document_anonymizer_memory.register_custom_patterns(st.session_state.custom_patterns)

    # Ensure custom faker operators from session state are initialized
    if use_custom_faker_operators and 'custom_faker_operators' in st.session_state:
        document_anonymizer_memory.initialize_faker_operators(detected_language, st.session_state.custom_faker_operators)

    chatbot_memory = ChatbotMemory(document_anonymizer_memory, document_content, openai_api_key)
    return chatbot_memory

# Create a container for the chatbot
chatbot_container = st.container()

with chatbot_container:
    # Check if 'start_chatbot' exists in session state, if not, initialize it
    if 'start_chatbot' not in st.session_state:
        st.session_state.start_chatbot = False

    # Button to start the chatbot
    if st.button("Start Chatbot"):
        st.session_state.start_chatbot = True

    # If the "Start Chatbot" button has been clicked, display the chatbot interface
    if st.session_state.start_chatbot:

        chatbot_memory = initialize_chatbot(document, openai_api_key, st.session_state.reset_counter)

        # Button to reset the chat
        if st.button("Reset Chat"):
            st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]
            st.session_state.reset_counter += 1  # Increment the reset_counter

        # Initialize chat messages if not present
        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

        # Display chat messages
        for msg in st.session_state.messages:
            if msg["role"] == "assistant":
                st.chat_message("assistant").write(msg["content"])
            else:
                st.chat_message("user").write(msg["content"])

        # Get user input and display chatbot's response
        if prompt := st.chat_input():
            if not openai_api_key:
                st.info("Please add your OpenAI API key to continue.")
                st.stop()

            # Append user's message to session state
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)

            # Get chatbot's response using ChatbotMemory class
            chatbot_response = chatbot_memory.view_answer(prompt)
            st.session_state.messages.append({"role": "assistant", "content": chatbot_response})
            st.chat_message("assistant").write(chatbot_response)


        # Button to display chatbot's memory
        if st.button("View Chatbot Memory"):
            memory_content = chatbot_memory.get_memory_content()
            st.write("Chatbot Memory Content:")
            st.write(memory_content)

        # Button to display chatbot's anonymizer mapping
        if st.button("View Chatbot anonymizer mapping"):
            mapping_content = chatbot_memory.document_anonymizer.display_mapping()
            st.write("Chatbot Anonymizer Content:")
            st.write(mapping_content)
###------------------------------------------------------------------------------------------------------------###
st.write('\n')
st.write("---")
st.header("Project Conclusion:")
Project_Conclusion = """As a conclusion of this project, I have found that anonymizing one's data before it gets sent to a large language model can definitely improve data security, but to get the best results, one would have to spend a lot of time testing and tweaking the different patterns and faker operators.

I discovered that some personal data can still slip through, although in most cases all or nearly all of the data was anonymized before being sent to the LLM and stored in the vector database.

There were occasional issues with the mapping when trying to deanonymize the chatbot's responses, indicating there is still room for improvement in the reversible anonymization process.

Overall, this project demonstrated that anonymization is a useful technique to balance utility and privacy when applying AI chatbots and other natural language systems. 

With refinement of the entity recognition and fake data generation, anonymizing chatbots could serve as an important safeguard for handling personal and sensitive information going forward.

If you want to know more about anonymizing data and how to integrate it into your system, write me a email. 

You can fine more intomation about the examples I have used at this link: 

https://python.langchain.com/docs/guides/privacy/presidio_data_anonymization/qa_privacy_protection
"""

st.write(Project_Conclusion)

st.write('\n')
st.write("---")