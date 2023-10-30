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

# Chatbot interface: 

st.title("💬 Single Chain Chatbot")


# Introduction
st.write("""
Welcome to the Text Interpreter Chatbot! This specialized tool is designed to analyze and interpret textual inputs, 
providing insights into the sentiment, subject, and potential price mentioned in the text.

**Use Cases:**
- **Sentiment Analysis:** Understand whether a piece of text conveys a positive, neutral, or negative sentiment.
- **Subject Identification:** Identify the main subject or topic of a given text.
- **Price Extraction:** Extract any mentioned price from the text, useful for quickly identifying product or service costs.

Whether you're analyzing customer reviews, product descriptions, or any text snippet, this chatbot can provide quick 
and valuable insights. Simply input your text and let the chatbot do the rest!

*Note: This is just one of the many chatbots available on this page. Feel free to explore others for different functionalities.*
""")


class TextInterpreter_SingleChain:
    def __init__(self, openai_key):
        # Define the ResponseSchema for each expected output
        self.response_schemas = [
            ResponseSchema(name="sentiment", description="Is the text positive, neutral or negative? Only provide these words"),
            ResponseSchema(name="subject", description="What subject is the text about? Use exactly one word."),
            ResponseSchema(name="price", description="How expensive was the product? Use None if no price was provided in the text")
        ]

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
        self.chat = ChatOpenAI(model='gpt-3.5-turbo-0613', openai_api_key=openai_key, temperature=0)

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


# Usage
interpreter_SingleChain = TextInterpreter_SingleChain(openai_api_key)

# Streamlit app
def main():
    st.title("Chatbot Integration with Streamlit")
    
    # Default text
    default_text = "The new iPhone 13 costs $999 and it's absolutely amazing with its camera features!"
    
    # Checkbox to insert default text
    if st.checkbox("Use default text", value=False):
        user_input = st.text_area("Enter your text here:", value=default_text)
    else:
        user_input = st.text_area("Enter your text here:")
    
    # When the user clicks the 'Interpret' button
    if st.button("Interpret"):
        # Get the chatbot's response
        response = interpreter_SingleChain.interpret(user_input)
        
        # Display the response
        st.write("Sentiment:", response.get("sentiment", "N/A"))
        st.write("Subject:", response.get("subject", "N/A"))
        st.write("Price:", response.get("price", "N/A"))

if __name__ == "__main__":
    main()