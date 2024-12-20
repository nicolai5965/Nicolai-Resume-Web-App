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
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.tools import Tool
from langchain_community.chat_message_histories import Neo4jChatMessageHistory
from langchain_community.graphs import Neo4jGraph
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain import hub
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain

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
    label="Download Resume 👈",
    data=PDFbyte_CV,
    file_name=resume_file,
    mime="application/octet-stream",
)

st.sidebar.write("---")

# Add pre-made questions
st.sidebar.write("### Pre-made Questions")

pre_made_questions = [
    "What is the plot of Matrix?",
    "Who directed The Matrix?",
    "List movies starring Leonardo DiCaprio.",
    "Find movies directed by Christopher Nolan.",
    "Tell me about the movie 'Interstellar'."
]

selected_question = st.sidebar.selectbox("Select a pre-made question:", pre_made_questions)

st.sidebar.write("Copy and paste the question into the input box below:")

st.sidebar.text_area("Question:", selected_question, height=100)


###------------------------------------------------------------------------------------------------------------###
### Section 1: Initialize Neo4jGraph and Embeddings ###
# Initialize Neo4jGraph
graph = Neo4jGraph(
    url=st.secrets["NEO4J_URI"],
    username=st.secrets["NEO4J_USERNAME"],
    password=st.secrets["NEO4J_PASSWORD"],
)

# Create the Embedding model
embeddings = OpenAIEmbeddings(
    openai_api_key=st.secrets["OPENAI_API_KEY"],
    model="text-embedding-3-small" #text-embedding-ada-002
)

# Verify connection using a simple query
try:
    result = graph.query("RETURN 1")
    connection_status = "Connected to Neo4j database!"
except Exception as e:
    connection_status = f"Error connecting to Neo4j database: {e}, pleace contact Nicolai if you want to see the project work. This is only a demo!"

###---------------------------------------------------------------------------------------------------------
### Section 2: Define LLMHandler Class ###

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
            default_model_name = 'claude-3-haiku-20240307'
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

###---------------------------------------------------------------------------------------------------------
### Section 3: Define Utility Functions ###

# Define memory using Neo4jChatMessageHistory and the Neo4jGraph
def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

def write_message(role, content):
    """
    Helper function to save a message to the session state.
    """
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    st.session_state.messages.append({"role": role, "content": content})

def get_session_id():
    ctx = get_script_run_ctx()
    if ctx is None:
        return None
    return ctx.session_id

def generate_response(user_input):
    """
    Generate a response using the conversational agent.
    """
    try:
        response = chat_agent.invoke(
            {"input": user_input},
            {"configurable": {"session_id": get_session_id()}}
        )
        return response['output']
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return "Sorry, I couldn't process your request."

def handle_submit():
    """
    Submit handler to process user input and display assistant response.
    """
    user_input = st.session_state.get('user_input', '')
    if user_input:
        # Clear previous retrieved context
        st.session_state['retrieved_context'] = []
        # Save user message
        write_message('user', user_input)

        # Generate assistant response
        with st.spinner('Thinking...'):
            response = generate_response(user_input)
            # Save assistant message
            write_message('assistant', response)

        # Clear the input box
        st.session_state.user_input = ''

###---------------------------------------------------------------------------------------------------------
### Section 4: Initialize Language Model Handler ###

# Initialize the language model handler
llm = LLMHandler("openai")

###---------------------------------------------------------------------------------------------------------
### Section 4: Create Tools ###

def create_movie_plot_tool(embeddings, graph, llm):
    """
    Creates the 'Movie Plot Search' tool for the agent.

    Parameters:
    - embeddings: The embedding model to use.
    - graph: The Neo4j graph instance.
    - llm: The language model handler instance.

    Returns:
    - Tool: The 'Movie Plot Search' tool to add to the agent's tools list.
    """
    # Initialize the Neo4jVector
    neo4jvector = Neo4jVector.from_existing_index(
        embeddings,                              # (1)
        graph=graph,                             # (2)
        index_name="moviePlots",                 # (3)
        node_label="Movie",                      # (4)
        text_node_property="plot",               # (5)
        embedding_node_property="plotEmbedding", # (6)
        retrieval_query="""
RETURN
    node.plot AS text,
    score,
    {
        title: node.title,
        directors: [ (person)-[:DIRECTED]->(node) | person.name ],
        actors: [ (person)-[r:ACTED_IN]->(node) | [person.name, r.role] ],
        tmdbId: node.tmdbId,
        source: 'https://www.themoviedb.org/movie/'+ node.tmdbId
    } AS metadata
"""
    )

    # Create the retriever
    retriever = neo4jvector.as_retriever()

    # Define the prompt instructions
    instructions = (
        "Use the given context to answer the question. "
        "If you don't know the answer, say you don't know. "
        "Context: {context}"
    )

    # Create the chat prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", instructions),
            ("human", "{input}"),
        ]
    )

    # Create the question-answer chain
    question_answer_chain = create_stuff_documents_chain(llm.language_model, prompt)
    plot_retriever = create_retrieval_chain(
        retriever,
        question_answer_chain
    )

    # Define the function that will be used by the tool
    def get_movie_plot(input):
        response = plot_retriever.invoke({"input": input})
        context_documents = response.get('context', [])
        # Store context documents in session state to display later
        st.session_state['retrieved_context'] = context_documents
        return response

    # Create and return the Tool object
    movie_plot_tool = Tool.from_function(
        name="Movie Plot Search",
        description="For when you need to find information about movies based on a plot",
        func=get_movie_plot,
    )

    return movie_plot_tool

###-------- General Chat Tool
# Define the chat prompt
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a movie expert providing information about movies."),
        ("human", "{input}"),
    ]
)
# Create the movie chat chain
movie_chat = chat_prompt | llm.language_model | StrOutputParser()

###-------- Text to Cypher Tool

CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer translating user questions into Cypher to answer questions about movies and provide recommendations.
Convert the user's question based on the schema.

Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

Do not return entire nodes or embedding properties.

Fine Tuning:

For movie titles that begin with "The", move "the" to the end. For example "The 39 Steps" becomes "39 Steps, The" or "the matrix" becomes "Matrix, The".


Schema:
{schema}

Question:
{question}

Cypher Query:
"""

cypher_prompt = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)


cypher_qa = GraphCypherQAChain.from_llm(
    llm.language_model,
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True,
    cypher_prompt=cypher_prompt
)



###---------------------------------------------------------------------------------------------------------
### Section 6: Create Tools List ###

# Call the function to create movie_plot_tool
movie_plot_tool = create_movie_plot_tool(embeddings, graph, llm)

# Define tools including movie_plot_tool
tools = [
    Tool.from_function(
        name="General Chat",
        description="For general movie chat not covered by other tools",
        func=movie_chat.invoke,
    ),
    movie_plot_tool,
    Tool.from_function(
        name="Movie information",
        description="Provide information about movies questions using Cypher",
        func = cypher_qa
    )
]

###---------------------------------------------------------------------------------------------------------
### Section 7: Create Conversational Agent ###
# Define the agent prompt
# agent_prompt = hub.pull("hwchase17/react-chat") #old pre made prompt

agent_prompt = PromptTemplate.from_template("""
You are a movie expert providing information about movies.
Be as helpful as possible and return as much information as possible.
Do not answer any questions that do not relate to movies, actors or directors.

Do not answer any questions using your pre-trained knowledge, only use the information provided in the context.

TOOLS:
------

You have access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
""")

# Create the agent
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


###---------------------------------------------------------------------------------------------------------
### Section 8: Streamlit User Interface ###

# Streamlit UI
st.title("GraphDB Conversational Explorer")


# Project Description
project_description = """
**Project Title:** Neo4j-Backed Movie Recommendation Chatbot

---

**Project Description:**

I developed a sophisticated chatbot that provides movie recommendations and information by leveraging a Neo4j graph database, Langchain, and Streamlit. This project involved creating an interactive chat interface where users can ask complex questions about movies, actors, and directors. The chatbot intelligently interprets user queries and retrieves accurate information from the Neo4j database, enriched with the capabilities of advanced language models like OpenAI's GPT-4 and Anthropic's Claude.

**Key Features:**

- **Interactive Chat Interface:** Built with Streamlit, the interface allows users to engage in natural language conversations with the chatbot.
- **Complex Query Handling:** Capable of understanding and answering intricate questions such as:
  - *"How many movies has Tom Hanks acted in?"*
  - *"What is the most popular movie in the database?"*
  - *"Can you recommend a movie for fans of The Matrix and Casino?"*
- **Conversational Memory:** Utilizes Neo4j to maintain conversation history, allowing for context-aware interactions.
- **Retrieval Augmented Generation (RAG):** Implements RAG to enhance the chatbot's responses with relevant information from the database.
- **Custom Tools and Agents:** Developed specialized tools for movie plot searches and Cypher query generation to interact with the Neo4j database effectively.

**Skills Demonstrated:**

- **Python Programming:** Extensive use of Python for backend development and integration of various libraries and APIs.
- **Graph Databases (Neo4j):** Proficient in setting up, querying, and managing data within a Neo4j graph database.
- **Natural Language Processing (NLP):** Integrated language models to understand and generate human-like responses.
- **Web Development with Streamlit:** Created a user-friendly web interface for seamless interaction with the chatbot.
- **API Integration:** Worked with OpenAI and Anthropic APIs to incorporate advanced language models.
- **Data Retrieval and Processing:** Implemented techniques like RAG and developed custom retrieval functions.
- **Conversational AI:** Built agents with memory and context handling capabilities using Langchain.

**Technologies and Tools Used:**

- **Programming Language:** Python
- **Databases:** Neo4j Graph Database
- **Frameworks and Libraries:**
  - **Langchain:** For building chains and agents that interact with language models.
  - **OpenAI API:** Accessed GPT-4 models for language understanding.
  - **Anthropic API:** Utilized Claude models for advanced NLP tasks.
  - **Streamlit:** Developed the front-end chat interface.
  - **Py2neo and Neo4j Drivers:** For connecting and executing queries on the Neo4j database.
- **AI Models:** GPT-4, Claude
- **Additional Libraries:**
  - **Neo4jGraph and Neo4jVector:** For graph data handling and vector storage.
  - **Chat Message Histories:** Managed conversation histories within Neo4j.
  - **Prompt Templates:** Crafted custom prompts for language models.

**Project Outcomes:**

- Successfully created a functional chatbot that can handle complex movie-related queries.
- Demonstrated the integration of a graph database with state-of-the-art language models.
- Showcased the ability to build full-stack applications involving database management, backend logic, and frontend interfaces.

---

This project highlights my capability to develop complex AI-driven applications by integrating multiple technologies. It reflects my proficiency in Python, database management with Neo4j, natural language processing, and web development with Streamlit. Through this project, I have demonstrated strong problem-solving skills and the ability to create user-centric solutions that leverage advanced AI models.
"""

# Use an expander to hide the detailed description
with st.expander("Read more about this project"):
    st.markdown(project_description)
    # Display the architecture diagram inside the expander
    image_path = "pages/Images/neo4j_chatbot_architecture.png"
    st.image(image_path, caption="Architecture Diagram of the Neo4j-Backed Chatbot", use_column_width=True)


# User input
st.text_input("You:", key='user_input', on_change=handle_submit)

st.write(connection_status )

# Chat interface
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Render messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# After rendering messages, display the retrieved context
if 'retrieved_context' in st.session_state and st.session_state['retrieved_context']:
    with st.expander("Show Retrieved Context"):
        context_documents = st.session_state['retrieved_context']
        # Display the context documents using tabs
        if context_documents:
            # Create a tab for each document
            tab_titles = [f"Document {idx}" for idx in range(1, len(context_documents) + 1)]
            tabs = st.tabs(tab_titles)
            for idx, (tab, doc) in enumerate(zip(tabs, context_documents), start=1):
                with tab:
                    st.write(f"### Document {idx}")
                    st.write(f"**Title:** {doc.metadata.get('title', 'N/A')}")
                    st.write(f"**Text:** {doc.page_content}")
                    st.write(f"**Metadata:**")
                    for key, value in doc.metadata.items():
                        st.write(f"- **{key.capitalize()}:** {value}")
        else:
            st.write("No context documents retrieved.")
###------------------------------------------------------------------------------------------------------------###




