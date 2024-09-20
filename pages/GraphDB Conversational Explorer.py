###------------------------------------------------------------------------------------------------------------###
#Importing libraries

import os
import uuid
import datetime
import streamlit as st

from neo4j import GraphDatabase
from py2neo import Graph

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.tools import Tool
from langchain_community.chat_message_histories import Neo4jChatMessageHistory
from langchain_community.graphs import Neo4jGraph
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain import hub
from streamlit.runtime.scriptrunner import get_script_run_ctx

# Set environment variables using Streamlit secrets
os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]
os.environ['LANGCHAIN_API_KEY'] = st.secrets["LANGCHAIN_API_KEY"]
os.environ['ANTHROPIC_API_KEY'] = st.secrets["ANTHROPIC_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "neo4j_learning"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# Configure variables for Neo4j access
NEO4J_URI = st.secrets["NEO4J_URI"]
NEO4J_USERNAME = st.secrets["NEO4J_USERNAME"]
NEO4J_PASSWORD = st.secrets["NEO4J_PASSWORD"]


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
# Initialize Neo4jGraph
graph = Neo4jGraph(
    url=st.secrets["NEO4J_URI"],
    username=st.secrets["NEO4J_USERNAME"],
    password=st.secrets["NEO4J_PASSWORD"],
)

# Verify connection using a simple query
try:
    result = graph.query("RETURN 1")
    connection_status = "Connected to Neo4j database!"
except Exception as e:
    connection_status = f"Error connecting to Neo4j database: {e}"

# Streamlit UI
st.title("GraphDB Conversational Explorer")

st.write(connection_status)  # Display connection status below the title

class LLMHandler:
    """
    A handler for managing language models from different providers.
    """
    def __init__(self, llm_provider, max_tokens=300, temperature=0.2, model_name=None):
        self.llm_provider = llm_provider
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Common metadata and tags
        self.common_metadata = {
            "session_id": str(uuid.uuid4()),
            "timestamp": datetime.datetime.now().isoformat(),
            "model_provider": self.llm_provider
        }
        self.common_tags = ["user_interaction", "query_handling", self.llm_provider]

        # Set default model names if not provided
        if self.llm_provider == "openai":
            default_model_name = 'gpt-4o-mini-2024-07-18'
            model_name = model_name or default_model_name

            # Initialize OpenAI's Chat model
            self.language_model = ChatOpenAI(
                model_name=model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                tags=self.common_tags,
                metadata=self.common_metadata,
                name="CustomChainName"
            )
        elif self.llm_provider == "anthropic":
            default_model_name = 'claude-2'
            model_name = model_name or default_model_name

            # Initialize Anthropic's Chat model
            self.language_model = ChatAnthropic(
                model=model_name,
                max_tokens_to_sample=self.max_tokens,
                temperature=self.temperature,
                tags=self.common_tags,
                metadata=self.common_metadata,
                name="CustomChainName"
            )
        else:
            raise ValueError(f"Invalid llm_provider '{llm_provider}'. Must be either 'openai' or 'anthropic'.")

    def show_settings(self):
        """
        Display the current settings of the language model handler.
        """
        # Access the model name attribute
        if hasattr(self.language_model, 'model_name'):
            model_name = self.language_model.model_name
        elif hasattr(self.language_model, 'model'):
            model_name = self.language_model.model
        else:
            model_name = None

        # Access the max_tokens attribute
        if hasattr(self.language_model, 'max_tokens'):
            max_tokens = self.language_model.max_tokens
        elif hasattr(self.language_model, 'max_tokens_to_sample'):
            max_tokens = self.language_model.max_tokens_to_sample
        else:
            max_tokens = self.max_tokens

        settings = {
            "llm_provider": self.llm_provider,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "common_metadata": self.common_metadata,
            "common_tags": self.common_tags,
            "language_model": {
                "model_name": model_name,
                "max_tokens": max_tokens,
                "temperature": self.language_model.temperature,
                "tags": self.language_model.tags,
                "metadata": self.language_model.metadata,
                "name": self.language_model.name
            }
        }
        return settings

# Initialize the language model handler
llm = LLMHandler("openai")

# Define the chat prompt
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a movie expert providing information about movies."),
        ("human", "{input}"),
    ]
)

# Create the movie chat chain
movie_chat = chat_prompt | llm.language_model | StrOutputParser()

# Define tools
tools = [
    Tool.from_function(
        name="General Chat",
        description="For general movie chat not covered by other tools",
        func=movie_chat.invoke,
    )
]

# Define memory using Neo4jChatMessageHistory and the Neo4jGraph
def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

# Create the agent
agent_prompt = hub.pull("hwchase17/react-chat")
agent = create_react_agent(llm.language_model, tools, agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

# Create the chat agent with message history
chat_agent = RunnableWithMessageHistory(
    agent_executor,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

def write_message(role, content, save=True):
    """
    Helper function to save a message to the session state and write it to the UI.
    """
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if save:
        st.session_state.messages.append({"role": role, "content": content})

    # with st.chat_message(role):
    #     st.markdown(content)

def get_session_id():
    ctx = get_script_run_ctx()
    if ctx is None:
        return None
    return ctx.session_id

def generate_response(user_input):
    """
    Generate a response using the conversational agent.
    """
    response = chat_agent.invoke(
        {"input": user_input},
        {"configurable": {"session_id": get_session_id()}}
    )
    return response['output']

# Handle user input and generate responses
def handle_submit():
    """
    Submit handler to process user input and display assistant response.
    """
    user_input = st.session_state.get('user_input', '')
    if user_input:
        # Save user message
        write_message('user', user_input)

        # Generate and save assistant response
        with st.spinner('Thinking...'):
            response = generate_response(user_input)
            write_message('assistant', response)

        # Clear the input box
        st.session_state.user_input = ''


# Chat interface
if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    write_message(message["role"], message["content"], save=False)

# User input
st.text_input("You:", key='user_input', on_change=handle_submit)
