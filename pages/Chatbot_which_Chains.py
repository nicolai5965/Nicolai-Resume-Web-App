###------------------------------------------------------------------------------------------------------------###
#Importing libraries
from pathlib import Path
import os
import streamlit as st
import re
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.chains.router import MultiPromptChain
from langchain.chains.llm import LLMChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
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

openai_api_key = os.environ.get('OPENAI_API_KEY', None)
###------------------------------------------------------------------------------------------------------------###
### Sidebar
st.sidebar.write('\n')
resume_file = "pages/Nicolai's Resume.pdf"
LinkedIn_link = '[My LinkedIn](https://www.linkedin.com/in/nicolai-s%C3%B8derberg-907680238/)'
st.sidebar.markdown(LinkedIn_link, unsafe_allow_html=True)

GitHub_link = '[My GitHub repo](https://github.com/nicolai5965)'
st.sidebar.markdown(GitHub_link, unsafe_allow_html=True)

with open(resume_file, "rb") as pdf_file:
    PDFbyte_CV = pdf_file.read()
    
st.sidebar.download_button(
    label = "Download Resume 👈",
    data = PDFbyte_CV,
    file_name = resume_file,
    mime ="application/octet-stream",)

st.sidebar.write("---")

###------------------------------------------------------------------------------------------------------------###
# Single chain chatbot code 
class TextInterpreter_SingleChain:
    def __init__(self, openai_key, response_schemas=None):
        # If no custom schemas are provided, use the default ones
        if response_schemas is None:
            self.response_schemas = [
                ResponseSchema(name="sentiment", description="Is the text positive, neutral or negative? Only provide these words"),
                ResponseSchema(name="subject", description="What subject is the text about? Use exactly one word."),
                ResponseSchema(name="price", description="How expensive was the product? Use None if no price was provided in the text")
            ]
        else:
            self.response_schemas = response_schemas

        # Create a StructuredOutputParser using the defined schemas
        self.parser = StructuredOutputParser.from_response_schemas(self.response_schemas)

        # Retrieve the format instructions from the parser
        format_instructions = self.parser.get_format_instructions()

        # Define the template using the format instructions
        self.template = f"""

        Just return the JSON, do not add ANYTHING, NO INTERPRETATION!

        text: {{input}}

        {{format_instructions}}
        """

        # Initialize LangChain's OpenAI model with the provided API key
        self.chat = ChatOpenAI(model='gpt-4o-mini-2024-07-18', openai_api_key=openai_key, temperature=0)

    def interpret(self, text):
        # Create a ChatPromptTemplate using the template
        prompt = ChatPromptTemplate.from_template(template=self.template)

        # Format the messages using the format_messages() method
        messages = prompt.format_messages(input=text, format_instructions=self.parser.get_format_instructions())

        # Get the response using the ChatOpenAI model
        response = self.chat(messages)

        # Parse the response content to get the structured output
        output_dict = self.parser.parse(response.content)

        return output_dict

    def print_settings(self):
        # Print the default settings (attributes) of the ChatOpenAI instance
        for attribute, value in self.chat.__dict__.items():
            print(f"{attribute}: {value}")

###------------------------------------------------------------------------------------------------------------###
# Multi chain chatbot:

class TextInterpreter_MultiChain:
    def __init__(self, openai_key, word_limit=50):
        self.word_limit = word_limit

        # Initialize LangChain's OpenAI model with the provided API key
        self.chat = ChatOpenAI(model='gpt-3.5-turbo-0613', openai_api_key=openai_key, temperature=0)

        # Initialize chains
        self._initialize_chains()

    def _get_concise_instruction(self):
        return f"Provide a concise answer, ideally within {self.word_limit} words, ensuring it's complete and makes sense."

    def _initialize_chains(self):
        self.chain_sentiment = LLMChain(
            llm=self.chat,
            prompt=PromptTemplate.from_template("Determine the sentiment of the review: {review_text}. (positive, negative, neutral)"),
            output_key="sentiment"
        )

        self.chain_subject = LLMChain(
            llm=self.chat,
            prompt=PromptTemplate.from_template("Identify the main subject of the review: {review_text}. (one word)"),
            output_key="subject"
        )

        self.chain_price = LLMChain(
            llm=self.chat,
            prompt=PromptTemplate.from_template("Extract the price mentioned in the review: {review_text}. (None if no price mentioned)"),
            output_key="price"
        )

        concise_instruction = self._get_concise_instruction()
        self.chain_review = LLMChain(
            llm=self.chat,
            prompt=PromptTemplate.from_template(f"Analyze the sentiment of the following review: {{review_text}}. {concise_instruction}"),
            output_key="detailed_sentiment"
        )

        self.chain_comment = LLMChain(
            llm=self.chat,
            prompt=PromptTemplate.from_template(f"Given the sentiment of the review: {{detailed_sentiment}}. {concise_instruction}"),
            output_key="comment"
        )

        self.chain_follow_up = LLMChain(
            llm=self.chat,
            prompt=PromptTemplate.from_template(f"Write a follow-up comment on the {{review_text}}. {concise_instruction}"),
            output_key="follow-up"
        )

        self.chain_summary = LLMChain(
            llm=self.chat,
            prompt=PromptTemplate.from_template(f"Summarise the {{review_text}} and our {{comment}} in one concise sentence. {concise_instruction}"),
            output_key="summary"
        )

        self.chain_improvements = LLMChain(
            llm=self.chat,
            prompt=PromptTemplate.from_template(f"From the review, suggest main improvements in one concise sentence. {concise_instruction}"),
            output_key="improvements"
        )

        self.overall_chain = SequentialChain(
            chains=[self.chain_sentiment, self.chain_subject, self.chain_price, self.chain_review, self.chain_comment, self.chain_summary, self.chain_improvements, self.chain_follow_up],
            input_variables=["review_text"],
            output_variables=["sentiment", "subject", "price", "detailed_sentiment", "comment", "summary", "improvements", "follow-up"]
        )

    def multi_chain_interpret(self, review_text):
        return self.overall_chain({"review_text": review_text})

    def print_settings(self):
        # Print the default settings (attributes) of the ChatOpenAI instance
        for attribute, value in self.chat.__dict__.items():
            print(f"{attribute}: {value}")


###------------------------------------------------------------------------------------------------------------###
# Single Chain Chatbot interface: 
st.title("💬 Single Chain Chatbot")

with st.expander("Single Chain Chatbot Introduction"):
    # Introduction
    st.write("""
    Welcome to the Single Chain Text Interpreter Chatbot! This specialized tool is designed to analyze and interpret textual inputs, 
    providing insights into the sentiment, subject, and potential price mentioned in the text.

    **Use Cases:**
    - **Sentiment Analysis:** Understand whether a piece of text conveys a positive, neutral, or negative sentiment.
    - **Subject Identification:** Identify the main subject or topic of a given text.
    - **Price Extraction:** Extract any mentioned price from the text, useful for quickly identifying product or service costs.

    Whether you're analyzing customer reviews, product descriptions, or any text snippet, this chatbot can provide quick 
    and valuable insights. Simply input your text and let the chatbot do the rest!

    *Note: This is just one of the many chatbots available on this page. Feel free to explore others for different functionalities.*
    """)

###-----------------------------------###
# Custom ResponseSchema definer
st.subheader("What is the ResponseSchema")

# Expander for ResponseSchema description
with st.expander("ResponseSchema Info"):
    st.write("""
    The ResponseSchema in Langchain defines the structure of the output that the agent should return.
    Specifically, the ResponseSchema allows you to specify multiple fields that the output should contain, along with a name and description for each field.
    """)

# Expander for 'Name' description
with st.expander("Name Info"):
    st.write("""
    The 'Name' field in the ResponseSchema represents the key or identifier for the extracted information.
    For example, if you want the chatbot to identify sentiments in the text, you might use 'sentiment' as the name.
    """)

# Expander for 'Description' description
with st.expander("Description Info"):
    st.write("""
    The 'Description' field provides a brief explanation or instruction about the information you want to extract.
    For instance, if you're identifying sentiments, the description might be 'Is the text positive, neutral, or negative?'.
    """)

# Basic Response Schemas
BASIC_SCHEMAS = [
    ResponseSchema(name="sentiment", description="Is the text positive, neutral or negative? Only provide these words"),
    ResponseSchema(name="subject", description="What subject is the text about? Use exactly one word."),
    ResponseSchema(name="price", description="How expensive was the product? Use None if no price was provided in the text")
]

# Advanced Response Schemas
ADVANCED_SCHEMAS = [
    ResponseSchema(name="Text recognition", description="Identify names, places, and organizations mentioned in the text."),
    ResponseSchema(name="Text topic", description="What are the main topics or themes discussed in the text?"),
]

###-----------------------------------###
st.subheader("Chatbot Integration with Streamlit")

# Checkbox for more schemas
use_advanced = st.checkbox("Use More Schemas")

# If the user wants to use advanced settings, combine both basic and advanced schemas
if use_advanced:
    schemas = BASIC_SCHEMAS + ADVANCED_SCHEMAS
else:
    schemas = BASIC_SCHEMAS

with st.expander("Selected ResponseSchemas:"):
    for schema in schemas:
        st.write(f"- {schema.name}: {schema.description}")


# Check if 'interpreter_SingleChain' exists in the session state
if 'interpreter_SingleChain' not in st.session_state:
    st.session_state.interpreter_SingleChain = None

# Button to initialize the chatbot with selected schemas
if st.button("Initialize Chatbot with Selected Schemas"):
    st.session_state.interpreter_SingleChain = TextInterpreter_SingleChain(openai_api_key, response_schemas=schemas)
    st.write("Chatbot initialized with selected ResponseSchemas!")


# Default text
default_text = """Apple recently unveiled the iPhone 13 at their California headquarters. 
Priced at $999, this latest model has garnered positive reviews for its advanced camera features and improved battery life. 
Many tech enthusiasts believe that Apple's focus on augmented reality and 5G capabilities will set new standards in the smartphone industry. 
Additionally, the collaboration with organizations like NASA for satellite communication features has piqued the interest of many. 
Overall, the iPhone 13 seems to be a promising step forward in mobile technology."""

# Checkbox to insert default text
if st.checkbox("Use default text", value=False):
    user_input = st.text_area("Enter your text here:", value=default_text)
else:
    user_input = st.text_area("Enter your text here:")

# When the user clicks the 'Interpret' button
if st.button("Interpret", key="interpret_button"):
    # Check if the chatbot has been initialized in the session state
    if st.session_state.interpreter_SingleChain:
        # Get the chatbot's response
        response = st.session_state.interpreter_SingleChain.interpret(user_input)
        
        # Display the response
        for key, value in response.items():
            st.write(f"{key.capitalize()}: {value or 'N/A'}")
    else:
        st.warning("Please initialize the chatbot with selected ResponseSchemas first!")

###------------------------------------------------------------------------------------------------------------###



# Streamlit UI
st.title("Multi Chained Text Interpreter Chatbot")

# Introduction Expander
with st.expander("Introduction"):
    st.write("""
    In the rapidly evolving world of Natural Language Processing (NLP), the ability to extract and interpret nuanced information from textual data is paramount. Enter the Multi Chained Text Interpreter Chatbot—a cutting-edge solution designed to delve deep into the layers of textual reviews and provide comprehensive insights.
    """)

# What is the Multi Chained Text Interpreter Chatbot? Expander
with st.expander("What is the Multi Chained Text Interpreter Chatbot?"):
    st.write("""
    The Multi Chained Text Interpreter Chatbot is an advanced NLP tool that leverages the power of OpenAI's GPT-3.5 Turbo model to analyze and interpret textual reviews. Instead of providing a singular output, this chatbot employs a multi-chain approach, breaking down the review into various components such as sentiment, subject, price, and more.
    """)

# Key Features Expander
with st.expander("Key Features"):
    st.write("""
    1. **Sequential Analysis:** The chatbot processes reviews in a sequential manner, extracting multiple pieces of information one after the other, ensuring a thorough analysis.
    2. **Concise Outputs:** With a set word limit, the chatbot ensures that the responses are concise yet informative, making it easier for users to grasp the insights.
    3. **Versatility:** From determining the sentiment of a review to suggesting improvements and providing follow-up comments, this chatbot covers a wide spectrum of analytical capabilities.
    4. **User-Friendly Interface:** Integrated with Streamlit, the chatbot offers an intuitive interface where users can input reviews, choose to use default text, and instantly receive interpretations.
    """)

# How Does It Work? Expander
with st.expander("How Does It Work?"):
    st.write("""
    At its core, the chatbot utilizes a series of chains—each responsible for a specific type of analysis. For instance, one chain might determine the sentiment of a review, while another identifies the main subject. These chains work in tandem, feeding information from one to the next, culminating in a comprehensive interpretation of the input review.
    """)

# Applications Expander
with st.expander("Applications"):
    st.write("""
    The potential applications of the Multi Chained Text Interpreter Chatbot are vast. Businesses can use it to gain insights into customer feedback, researchers can analyze qualitative data, and individuals can get a deeper understanding of any textual content they come across.
    """)

# Conclusion Expander
with st.expander("Conclusion"):
    st.write("""
    In a world inundated with textual data, the Multi Chained Text Interpreter Chatbot stands out as a beacon of analytical prowess. Whether you're a business looking to understand customer sentiment or an individual curious about the nuances of a piece of text, this chatbot promises to deliver insights that are both deep and actionable.
    """)

# Checkbox to add default text
if st.checkbox("Use default text", key="Multi_Chain"):
    review_text = st.text_area("Enter your review:", value=default_text)
else:
    # Text input for user's review
    review_text = st.text_area("Enter your review:")

if st.button("Interpret"):
    st.write("It take a second")
    interpreter_MultiChain = TextInterpreter_MultiChain(openai_api_key, word_limit=50)
    if review_text:
        # Get the results from the chatbot
        multi_chain_result = interpreter_MultiChain.multi_chain_interpret(review_text)
        
        # Display the results
        st.write("Sentiment:", multi_chain_result["sentiment"])
        st.write("Subject:", multi_chain_result["subject"])
        st.write("Price:", multi_chain_result["price"])
        st.write("Detailed Sentiment:", multi_chain_result["detailed_sentiment"])
        st.write("Comment:", multi_chain_result["comment"])
        st.write("Summary:", multi_chain_result["summary"])
        st.write("Improvements:", multi_chain_result["improvements"])
        st.write("Follow-up:", multi_chain_result["follow-up"])
    else:
        st.warning("Please enter a review to interpret.")
