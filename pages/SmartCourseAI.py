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
# Add the header
st.header("Welcome to the SmartCourseAI Feedback Assistant!")

# Define the introduction text as a variable
introduction_text = """
This app leverages advanced language models to provide precise, actionable feedback on your course responses. By focusing strictly on the provided course material, it ensures that you receive guidance tailored to your learning needs. Here’s a comprehensive overview:

## What Does This App Do?
The SmartCourseAI Feedback Assistant is designed to:
- **Evaluate Responses**: Assess your answers to course questions based on a thorough understanding of the material.
- **Provide Ratings**: Offer a strict, accurate rating on a scale from 1 to 10 to reflect your comprehension.
- **Give Constructive Feedback**: Deliver detailed, actionable feedback to help you improve.

## Key Features:
- **Initial Feedback**: Get immediate evaluation and feedback based on your submitted answers, adhering strictly to the course material.
- **Reflection and Correction**: The app employs a rigorous reflection and correction process, refining feedback to ensure accuracy and completeness.
- **Aggregated Feedback**: At the end of your course, receive a comprehensive summary of your performance and feedback, highlighting areas for improvement and reinforcing strengths.

## How Does It Work?
- **Language Model Integration**: Utilizing state-of-the-art models such as OpenAI and Anthropic, the system processes and evaluates your responses with high precision.
- **Dynamic and Strict Prompts**: The system generates prompts that strictly adhere to the course guidelines, ensuring that feedback is both relevant and constructive.
- **Iterative Process**: Engage in a process that involves multiple stages of feedback refinement, enhancing your learning through continuous improvement.

## Behavior Guidelines for Language Models:
The language models follow stringent behavior guidelines to ensure the quality and relevance of feedback:
- **Language and Specialization**: At the moment, the models communicate only in English and are specialized in providing guidance strictly based on the course material, although the language can be changed in the future.
- **Strict Rating Criteria**: Ratings are given on a scale of 1 to 10, with strict adherence to the rating guidelines to ensure only those who truly understand the material pass. The rating criteria and LLM behaviors can be changed according to the course's meaning.
- **Feedback Format**: Feedback is clear, concise, and directly related to the course material, ensuring you receive actionable insights.

## Getting Started:
1. **Submit Your Answers**: Answer the course questions as prompted.
2. **Receive Initial Feedback**: Review the immediate feedback provided.
3. **Engage in Reflection and Correction**: Participate in the iterative feedback process to refine your understanding.
4. **Review Final Summary**: At the end of the session, access a detailed summary of your performance, including ratings and comprehensive feedback.

By following these steps, you can enhance your understanding of the course material and improve your performance through targeted feedback. Let’s embark on this journey of learning and improvement together!
"""

# Initialize the session state if not already initialized
if 'show_introduction' not in st.session_state:
    st.session_state.show_introduction = False

# Add a button to toggle the introduction text
if st.button("Learn More About SmartCourseAI"):
    st.session_state.show_introduction = not st.session_state.show_introduction

# Display the introduction text if the session state is set to True
if st.session_state.show_introduction:
    st.write(introduction_text)


st.write('\n')
st.write("---")

# Define the text as a variable
why_choose_smartcourseai_text = """
### Benefits of Using SmartCourseAI

Traditional courses often rely on passive learning methods, such as reading materials and answering multiple-choice questions, which may not fully engage learners or address their individual needs. SmartCourseAI offers a more effective and interactive approach to learning and improvement. Here's why it's better:

#### Personalized and Constructive Feedback
Unlike traditional methods, our app provides detailed, personalized feedback on your answers. This helps you understand your strengths and pinpoint specific areas where you need improvement, rather than just indicating whether an answer is right or wrong.

#### Iterative Learning Process
With SmartCourseAI, learning is an ongoing process. The app engages you in multiple stages of feedback, reflection, and correction, ensuring a deeper understanding of the material. This iterative approach helps solidify your knowledge and improves retention.

#### Strict and Accurate Evaluation
The language models are designed to evaluate responses strictly according to the course material, ensuring that you truly understand the content. This rigorous evaluation method ensures that passing the course is a genuine achievement.

#### Interactive and Engaging
SmartCourseAI makes learning interactive by involving you in a dynamic feedback loop. This engagement keeps you actively involved in the learning process, making it more enjoyable and effective than passive reading or static quizzes.

#### Comprehensive Final Feedback
At the end of the course, you receive a detailed summary of your performance, including a final rating and comprehensive feedback. This summary helps you understand your overall progress and provides clear guidance for future improvement.

By using SmartCourseAI, you gain a deeper, more nuanced understanding of the course material, actively engage in your learning process, and receive constructive feedback tailored to your needs. This approach not only helps you pass the course but also ensures you truly grasp the concepts and can apply them effectively.
"""

# Initialize the session state if not already initialized
if 'show_why_choose' not in st.session_state:
    st.session_state.show_why_choose = False

# Add a button to toggle the text display
if st.button("Why Choose SmartCourseAI?"):
    st.session_state.show_why_choose = not st.session_state.show_why_choose

# Display the text if the session state is set to True
if st.session_state.show_why_choose:
    st.write(why_choose_smartcourseai_text)

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

def process_course_questions2(course_material_qa, course_material, initial_llm_behavior_guidelines_new, max_words=45, llm_model="anthropic", max_iterations=2, test_mode=False):
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
# Initialize session state for toggling course material display
if 'show_course_material' not in st.session_state:
    st.session_state.show_course_material = False

# Input parameters
max_words = st.number_input('Max Words', min_value=1, max_value=1000, value=45)
llm_model = st.selectbox('LLM Model', ['openai', 'anthropic'])
max_iterations = st.number_input('Max Iterations', min_value=1, max_value=10, value=2)

# Add a button to toggle the display of course material
if st.button("Show/Hide Course Material"):
    st.session_state.show_course_material = not st.session_state.show_course_material

# Display course material if toggled on
if st.session_state.show_course_material:
    st.write(course_material)


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

        user_answer = st.text_input(f"Your answer to question {iteration_count + 1}", key=f"user_answer_{iteration_count}")

        if st.button(f"Submit Answer {iteration_count + 1}", key=f"submit_answer_{iteration_count}"):
            try:
                # Process the question with your backend logic
                output = asyncio.run(run_feedback_assistant(course_material, question_text, initial_llm_behavior_guidelines_new, user_answer, max_words, llm_model, max_iterations))
                output_json = convert_to_json(output)

                # Extract feedback using your existing logic
                que_feed_track = QuestionFeedbackTracker()
                que_feed_track.parse_feedback_information(question_text, output_json)
                final_results = que_feed_track.get_results()

                st.write(f"Rating: {final_results[0]['rating']}")
                st.write(f"Feedback: {final_results[0]['feedback']}")

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

# Sample call to the process_course_questions function
course_material_qa = {
    '1': {'question': 'What is the capital of France?'},
    '2': {'question': 'Explain the theory of relativity.'},
}


process_course_questions(course_material_qa, course_material, initial_llm_behavior_guidelines_new, max_words, llm_model, max_iterations)


###------------------------------------------------------------------------------------------------------------###

# Path to the JSON file
def read_json_file(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        st.write(f"Error: {e}")
        return None

# Path to the JSON file
file_path = 'pages/SmartCourseAI_Files/course_answers_feedback.json'

# Read the JSON file
data = read_json_file(file_path)

# # Display the data
# if data is not None:
#     st.write("JSON data:")
#     st.write(json.dumps(data, indent=4))

# Define the Pydantic model for aggregated feedback
class AggregateFeedback(BaseModel):
    passed_or_failed: str = Field(description="If the rating is over 8 the course taker has passed, else failed.")
    total_rating: float = Field(description="The total rating of all responses divided by the amount of ratings, including the rating range")  # Aggregated total rating
    combined_feedback: str = Field(description="The summarized feedback from all responses, that the course taker will read at the end of the course. So this summarized feedback should guide the course taker in the future.")  # Aggregated feedback

# Define the FeedbackAggregator class
class FeedbackAggregator:
    def __init__(self, llm_provider):
        # Initialize the LLMHandler with the given provider
        self.llm = LLMHandler(llm_provider=llm_provider, max_tokens=2000, temperature=0.1)

        # Initialize the Pydantic parser with the AggregateFeedback model
        self.parser = PydanticOutputParser(pydantic_object=AggregateFeedback)

        # Define the prompt template for the LLM
        self.prompt = PromptTemplate(
            template="""Given the following final rating and feedback, provide an overall summary that integrates all feedback and guides the course taker for future improvement.
            \nFinal Rating: {final_rating}
            \nAll Feedback: {all_feedback}.
            \nPlease provide a final overall feedback for the course taker based on the above information.
            \n{format_instructions}\n""",
            input_variables=["final_rating", "all_feedback"],  # Variables for user query
            partial_variables={"format_instructions": self.parser.get_format_instructions()},  # Format instructions for the parser
        )

        # Combine the prompt, LLM, and parser into a processing chain
        self.chain = self.prompt | self.llm.language_model | self.parser

    def aggregate_feedback(self, summarized_rating, collected_feedback):
        return self.chain.invoke({"final_rating": summarized_rating, "all_feedback": collected_feedback})


st.title("Full course Feedback")

# Placeholder for the ttj_output
ttj_output = transform_to_json(data, max_rating=10)
st.write(ttj_output)
if st.button("Aggregate Feedback"):
    try:
        # Initialize the FeedbackAggregator
        aggregator = FeedbackAggregator(llm_provider="anthropic")

        # Get the final feedback
        final_feedback = aggregator.aggregate_feedback(ttj_output["summarized_rating"], ttj_output["collcted_feedback"]) 
        # Display the final feedback
        with st.container():
            st.header("Final Feedback")

            # Display the pass/fail status
            pass_fail_container = st.container()
            with pass_fail_container:
                pass_fail_label = st.markdown(f"<span style='font-weight: bold; font-size: 18px;'>Pass/Fail Status:</span>", unsafe_allow_html=True)
                if final_feedback.passed_or_failed == "Passed":
                    pass_fail_status = st.markdown(f"<span style='color: green; font-weight: bold;'>{final_feedback.passed_or_failed}</span>", unsafe_allow_html=True)
                else:
                    pass_fail_status = st.markdown(f"<span style='color: red; font-weight: bold;'>{final_feedback.passed_or_failed}</span>", unsafe_allow_html=True)

            # Display the total rating
            rating_container = st.container()
            with rating_container:
                st.markdown(f"<span style='font-weight: bold; font-size: 18px;'>Total Rating:</span>", unsafe_allow_html=True)
                st.markdown(f"<span style='font-size: 20px;'>{final_feedback.total_rating}</span>", unsafe_allow_html=True)

            # Display the combined feedback
            feedback_container = st.container()
            with feedback_container:
                st.markdown(f"<span style='font-weight: bold; font-size: 18px;'>Combined Feedback:</span>", unsafe_allow_html=True)
                st.markdown(f"<div style='background-color: #000000; padding: 20px; border-radius: 5px; font-size: 16px;'>{final_feedback.combined_feedback}</div>", unsafe_allow_html=True)

    except (ValueError, KeyError) as e:
        st.write(f"Error: {e}")


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






