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
###------------------------------------------------------------------------------------------------------------###
# Configuring environment variables for API access and project identification
os.environ['OPENAI_API_KEY'] = os.environ.get('OPENAI_API_KEY', None)
os.environ['LANGCHAIN_API_KEY'] = os.environ.get('LANGCHAIN_API_KEY', None)
os.environ['ANTHROPIC_API_KEY'] = os.environ.get('ANTHROPIC_API_KEY', None)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Company_Course_Teacher"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
###------------------------------------------------------------------------------------------------------------###

def read_and_exec_file(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        # Create a dictionary to execute the code within
        local_vars = {}
        # Parsing the file content into variables
        exec(content, {}, local_vars)
        # Display variable names created by exec
        #st.write("Variable names created by exec:")
        #for var_name in local_vars.keys():
        #    st.write(var_name)
        # Update the global namespace with the new variables
        globals().update(local_vars)
    except SyntaxError as e:
        st.write(f"SyntaxError: {e}")
    except Exception as e:
        st.write(f"Error: {e}")

# Call the function for the first file
read_and_exec_file('pages/SmartCourseAI_Files/Ai_Company_Course_LLM_Prompts.txt')

# Call the function for the second file
read_and_exec_file('pages/SmartCourseAI_Files/Course_Material_And_QA.txt')

###------------------------------------------------------------------------------------------------------------###

class LLMHandler:
    def __init__(self, llm_provider, max_tokens=300, temperature=0.2):
        # Initialize the language model handler
        self.llm_provider = llm_provider  # Store the provider name

        # Common metadata and tags
        common_metadata = {
            "session_id": str(uuid.uuid4()),
            "timestamp": datetime.datetime.now().isoformat(),
            "model_provider": self.llm_provider
        }
        common_tags = ["user_interaction", "query_handling", self.llm_provider]

        if llm_provider == "openai":
            # Use OpenAI's model with additional tags and metadata
            self.language_model = ChatOpenAI(
                model_name='gpt-3.5-turbo-0125',
                max_tokens=max_tokens,
                temperature=temperature,
                tags=common_tags,
                metadata=common_metadata,
                name="CustomChainName"
            )
        elif llm_provider == "anthropic":
            # Use Anthropic's model with additional tags and metadata
            self.language_model = ChatAnthropic(
                model_name="claude-3-haiku-20240307",
                max_tokens=max_tokens,
                temperature=temperature,
                tags=common_tags,
                metadata=common_metadata,
                name="CustomChainName"
            )
        else:
            # Raise error for invalid provider
            raise ValueError("Invalid llm_provider. Must be either 'openai' or 'anthropic'.")


###------------------------------------------------------------------------------------------------------------###

class PromptManager:
    def __init__(self):
        # Initialize the prompt attribute
        self.prompt = None

    def initial_llm_prompt(self, course_material, course_question, behavior_guidelines, max_words):
        # Method to set up the initial prompt with provided details
        self.prompt = ChatPromptTemplate(
            messages=[
                # System message with course details and guidelines
                SystemMessagePromptTemplate.from_template(
                    f"""

                    Here are your own behavior guidelines:
                    {behavior_guidelines}

                    Course Material:
                    {course_material}

                    Course Question:
                    {course_question}

                    Limit your feedback to a maximum of {max_words} words.
                    """
                ),
                # Placeholder for messages
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

    def reflection_llm_prompt(self, reflection_behavior_guidelines, course_material, course_question, behavior_guidelines, user_answer, max_words):
        # Method to set up the initial prompt with provided details
        self.prompt = ChatPromptTemplate(
            messages=[
                # System message with course details and guidelines
                SystemMessagePromptTemplate.from_template(
                    f"""

                    Here are your own behavior guidelines:
                    {reflection_behavior_guidelines}


                    Here is the context for your evaluation:

                    Course Question:
                    {course_question}

                    User's Answer:
                    {user_answer}

                    Original AI behavior guidelines:
                    {behavior_guidelines}

                    Course Material:
                    {course_material}

                    Limit your feedback to a maximum of {max_words} words.
                    """
                ),
                # Placeholder for messages
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

    def correction_llm_prompt(self, correction_behavior_guidelines, user_answer, max_words):
        # Method to set up the initial prompt with provided details
        self.prompt = ChatPromptTemplate(
            messages=[
                # System message with course details and guidelines
                SystemMessagePromptTemplate.from_template(
                    f"""

                    Here are your own behavior guidelines:
                    {correction_behavior_guidelines}


                    Here are the users answer:
                    {user_answer}

                    And how the

                    Limit your feedback to a maximum of {max_words} words.
                    """
                ),
                # Placeholder for messages
                MessagesPlaceholder(variable_name="messages"),
            ]
        )


###------------------------------------------------------------------------------------------------------------###

class FeedbackAssistant:
    def __init__(self, llm_handler: LLMHandler, reflection_prompt_template, correction_prompt_template, max_iterations: int):
        # Ensure iterations is an even number
        if max_iterations % 2 != 0:
            raise ValueError("The number of iterations must be an even number.")

        # Initialize the feedback assistant with a specific language model
        self.language_model = llm_handler.language_model
        self.prompt_manager = PromptManager()

        # Store the provided templates
        self.reflection_prompt_template = reflection_prompt_template
        self.correction_prompt_template = correction_prompt_template

        # Set the number of iterations
        self.max_iterations = max_iterations

        # Initialize the pipelines (they will be set later)
        self.reflect_pipeline = None
        self.correct_pipeline = None
        self.initial_feedback_pipeline = None

    def setup_initial_feedback_prompt(self, course_material, course_question, behavior_guidelines, max_words):
        # Use PromptManager to create the initial feedback prompt
        self.prompt_manager.initial_llm_prompt(
            course_material,
            course_question,
            behavior_guidelines,
            max_words
        )
        # Set up the initial feedback pipeline
        self.initial_feedback_pipeline = self.prompt_manager.prompt | self.language_model

    async def initial_feedback_node(self, state: Sequence[BaseMessage]) -> List[BaseMessage]:
        # Generate an initial feedback based on the given state and return it as a message list
        return [await self.initial_feedback_pipeline.ainvoke({"messages": state})]

    def setup_reflection_prompt(self, course_material, course_question, behavior_guidelines, user_answer, max_words):
        # Use PromptManager to create the reflection prompt
        self.prompt_manager.reflection_llm_prompt(
            self.reflection_prompt_template,
            course_material,
            course_question,
            behavior_guidelines,
            user_answer,
            max_words
        )
        # Set up the reflection pipeline
        self.reflect_pipeline = self.prompt_manager.prompt | self.language_model

    def setup_correction_prompt(self, correction_prompt_template, user_answer, max_words):
        # Use PromptManager to create the correction prompt
        self.prompt_manager.correction_llm_prompt(
            correction_prompt_template,
            user_answer,
            max_words
        )
        # Set up the correction pipeline
        self.correct_pipeline = self.prompt_manager.prompt | self.language_model

    async def reflection_node(self, state: Sequence[BaseMessage]) -> List[BaseMessage]:
        # Generate a reflection based on the given state and return it as a message list
        return [await self.reflect_pipeline.ainvoke({"messages": state})]

    async def correction_node(self, messages: Sequence[BaseMessage]) -> List[BaseMessage]:
        # Translate message types for correction and generate new feedback based on the reflection
        message_type_map = {"ai": HumanMessage, "human": AIMessage}
        translated_messages = [messages[0]] + [message_type_map[msg.type](content=msg.content) for msg in messages[1:]]
        result = await self.correct_pipeline.ainvoke({"messages": translated_messages})
        return [HumanMessage(content=result.content)]

    def check_continuation(self, state: List[BaseMessage]) -> str:
        # Determine if the process should end or continue based on the number of iterations
        if len(state) > self.max_iterations:  # End after the specified number of iterations
            return END
        return "reflect"

    async def run_initial_feedback_graph(self, initial_user_input: str) -> List[BaseMessage]:
        builder = MessageGraph()
        builder.add_node("initial_feedback", self.initial_feedback_node)
        builder.set_entry_point("initial_feedback")

        # Add an edge to the END node
        builder.add_edge("initial_feedback", END)

        graph = builder.compile()
        event = await graph.ainvoke([HumanMessage(content=initial_user_input)])
        return event

    async def execute_feedback_graph(self, course_material, course_question, behavior_guidelines, user_answer, max_words) -> List[str]:
        # Initialize the initial feedback prompt
        self.setup_initial_feedback_prompt(course_material, course_question, behavior_guidelines, max_words)

        # Generate initial feedback
        initial_feedback = await self.run_initial_feedback_graph(user_answer)

        # Initialize reflection and correction prompts
        self.setup_reflection_prompt(course_material, course_question, behavior_guidelines, user_answer, max_words)
        self.setup_correction_prompt(self.correction_prompt_template, user_answer, max_words)

        # Now proceed with the reflection and correction loop using the initial feedback
        builder = MessageGraph()
        builder.add_node("reflect", self.reflection_node)
        builder.add_node("correct", self.correction_node)
        builder.set_entry_point("reflect")

        # Define conditional transitions and the overall flow between nodes
        builder.add_conditional_edges("correct", self.check_continuation)
        builder.add_edge("reflect", "correct")

        # Compile and execute the graph, storing results in a list
        graph = builder.compile()
        results = []

        ai_message_content = next((msg.content for msg in initial_feedback if isinstance(msg, AIMessage)), "")
        initial_feedback_message = HumanMessage(content=ai_message_content)



        async for event in graph.astream([initial_feedback_message]):
            if isinstance(event, HumanMessage) or isinstance(event, AIMessage):
                results.append({
                    "type": event.__class__.__name__,
                    "content": event.content,
                    "metadata": event.response_metadata if isinstance(event, AIMessage) else {}
                })
            else:
                results.append({
                    "type": event.__class__.__name__,
                    "content": event
                })
        return initial_feedback, results


###------------------------------------------------------------------------------------------------------------###

async def run_feedback_assistant(course_material: str, course_question: str, behavior_guidelines: str, user_answer: str, max_words: int, llm_provider: str, max_iterations: int):
    # Initialize the LLMHandler
    llm_handler = LLMHandler(llm_provider=llm_provider, max_tokens=2000, temperature=0.1)

    # Instantiate FeedbackAssistant with specified model and prompts
    assistant = FeedbackAssistant(
        llm_handler=llm_handler,
        reflection_prompt_template=reflection_prompt_template_single_shot,
        correction_prompt_template=correction_prompt_template_single_shot,
        max_iterations=max_iterations
    )

    # Run the feedback graph and get results
    results = await assistant.execute_feedback_graph(course_material, course_question, behavior_guidelines, user_answer, max_words)
    return results


###------------------------------------------------------------------------------------------------------------###

# Convert to JSON

def convert_to_json(output):
    messages = []
    # Process the first part of the output
    for msg in output[0]:
        if 'usage_metadata' in msg.__dict__:
            messages.append({
                "type": "ai",
                "content": msg.content,
                "id": msg.id,
                "metadata": {
                    "response_metadata": msg.response_metadata,
                    "usage_metadata": msg.usage_metadata
                }
            })
        else:
            messages.append({
                "type": "human",
                "content": msg.content,
                "id": msg.id
            })

    # Process the second part of the output
    for update in output[1]:
        for key, msgs in update['content'].items():
            for msg in msgs:
                if key == 'reflect':
                    messages.append({
                        "type": "reflection",
                        "content": msg.content,
                        "id": msg.id,
                        "metadata": {
                            "response_metadata": msg.response_metadata,
                            "usage_metadata": msg.usage_metadata
                        }
                    })
                elif key == 'correct':
                    messages.append({
                        "type": "correction",
                        "content": msg.content,
                        "id": msg.id
                    })

    return json.dumps({"messages": messages}, indent=2)


###------------------------------------------------------------------------------------------------------------###

class QuestionFeedbackTracker:
    def __init__(self):
        self.results = []

    def parse_feedback_information(self, course_question, json_data):
        data = json.loads(json_data)
        last_correction_content = ""
        last_correction_rating = ""
        last_correction_feedback = ""
        human_content = ""

        for message in data["messages"]:
            if message["type"] == "correction":
                last_correction_content = message["content"]
                # Extract rating and feedback from the correction content
                lines = last_correction_content.split("\n\n")
                if len(lines) >= 2:
                    last_correction_rating = lines[0].replace("Rating: ", "")
                    last_correction_feedback = lines[1].replace("Feedback:\n", "")
            if message["type"] == "human":
                human_content = message["content"]

        self.results.append({
            "course_question": course_question,
            "answer": human_content,
            "rating": last_correction_rating,
            "feedback": last_correction_feedback
        })

    def get_results(self):
        return self.results


###------------------------------------------------------------------------------------------------------------###

def process_course_questions(course_material_qa, course_material, initial_llm_behavior_guidelines_new, max_words=45, llm_model="anthropic", max_iterations=2, test_mode=False):
    output_list = []
    json_output_list = []
    final_results_list = []

    iteration_count = 0
    max_test_iterations = 2 if test_mode else len(course_material_qa)

    for question_key, question_data in course_material_qa.items():
        if iteration_count >= max_test_iterations:
            break

        question_text = question_data['question']

        st.write(f"Question: {question_text}")
        user_answer = input("Your answer: ")

        try:
            # Process the question with your backend logic
            output = asyncio.run(run_feedback_assistant(course_material, question_text, initial_llm_behavior_guidelines_new, user_answer, max_words, llm_model, max_iterations))
            output_json = convert_to_json(output)

            # Extract feedback using your existing logic
            que_feed_track = QuestionFeedbackTracker()
            que_feed_track.parse_feedback_information(question_text, output_json)
            final_results = que_feed_track.get_results()

            st.write(final_results[0]['rating'])
            st.write(final_results[0]['feedback'])

            st.write('-----------------------------------------------------------------------')

            output_list.append(output)
            json_output_list.append(output_json)
            final_results_list.append(final_results)

            iteration_count += 1

        except KeyError as e:
            st.write(f"An error occurred while processing: {e}")
            # Optionally, continue to the next iteration or break based on your preference
            continue
        except TypeError as e:
            st.write(f"Type error during processing: {e}")
            # Handle other potential TypeErrors gracefully
            continue

    return output_list, json_output_list, final_results_list, iteration_count



###------------------------------------------------------------------------------------------------------------###

def transform_to_json(final_results_list, max_rating=10):
    """
    Transforms a list of dictionaries containing course questions, answers, ratings, and feedback
    into a dictionary with an average rating and aggregated feedback.

    Args:
    final_results_list (list): List of dictionaries containing course data.
    max_rating (int): The maximum rating scale.

    Returns:
    dict: Dictionary with the average rating and aggregated feedback.
    """
    # Initialize variables for calculating the average rating and collecting feedback
    total_rating = 0
    rating_count = 0
    feedback_list = []

    # Process the input list
    for question_index, question_list in enumerate(final_results_list, start=1):
        for item in question_list:
            # Extract and sum ratings
            rating = float(item['rating'].split('/')[0].replace(',', '.'))
            total_rating += rating
            rating_count += 1

            # Collect feedback with placeholder question names
            feedback_list.append({
                f"question_{question_index}": item['feedback']
            })

    # Calculate the average rating
    average_rating = total_rating / rating_count  # Floating-point division
    # Prepare the output dictionary with formatted rating and range
    output = {
        "summarized_rating": f"{average_rating:.1f}/{max_rating}",  # Format to one decimal place and include rating range
        "collcted_feedback": feedback_list,
        "rating_range": max_rating
    }

    # Return the dictionary
    return output

###------------------------------------------------------------------------------------------------------------###




###------------------------------------------------------------------------------------------------------------###




###------------------------------------------------------------------------------------------------------------###




###------------------------------------------------------------------------------------------------------------###




###------------------------------------------------------------------------------------------------------------###




###------------------------------------------------------------------------------------------------------------###




###------------------------------------------------------------------------------------------------------------###




###------------------------------------------------------------------------------------------------------------###




###------------------------------------------------------------------------------------------------------------###




###------------------------------------------------------------------------------------------------------------###




###------------------------------------------------------------------------------------------------------------###






