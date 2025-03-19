# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# mypy: disable-error-code="arg-type"
import os
import re
import operator
from typing import List, Tuple, Dict, Union

import google
import vertexai
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

# For advanced math
import sympy
from sympy.parsing.mathematica import parse_mathematica

from app.retrievers import get_compressor, get_retriever  # Ensure these are correct
from app.templates import format_docs, inspect_conversation_template, rag_template  # Ensure these are correct

EMBEDDING_MODEL = "text-embedding-005"
LOCATION = "us-central1"
LLM = "gemini-1.5-pro-002"  # Use a more powerful model for advanced math
EMBEDDING_COLUMN = "embedding"
TOP_K = 5

data_store_region = os.getenv("DATA_STORE_REGION", "us")
data_store_id = os.getenv("DATA_STORE_ID", "sample-datastore")

# Initialize Google Cloud and Vertex AI (with error handling)
credentials, project_id = google.auth.default()
try:
    vertexai.init(project=project_id, location=LOCATION)
    print(f"Vertex AI initialized successfully. Project: {project_id}, Location: {LOCATION}")
except Exception as e:
    print(f"ERROR: Failed to initialize Vertex AI: {e}")
    #  Optionally exit here, or raise the exception:
    #  raise  # This will stop the program.
    import sys
    sys.exit(1)

try:
    embedding = VertexAIEmbeddings(
        project=project_id, location=LOCATION, model_name=EMBEDDING_MODEL
    )
    print("VertexAIEmbeddings initialized successfully.")
except Exception as e:
    print(f"ERROR: Failed to initialize VertexAIEmbeddings: {e}")
    import sys
    sys.exit(1)

def direct_calculator(expression):
    """Handles basic arithmetic, extended with exponentiation."""
    expression = re.sub(r'\s+', '', expression)
    if not re.match(r'^[\d\+\-\*\/\(\)\.\^]+$', expression):  # Added ^
        return None

    try:
        # Handle parentheses
        if '(' in expression:
            start = expression.rfind('(')
            end = expression.find(')', start)
            if start != -1 and end != -1:
                inner_result = direct_calculator(expression[start+1:end])
                new_expression = expression[:start] + str(inner_result) + expression[end+1:]
                return direct_calculator(new_expression)

        # Handle exponentiation (^)
        if '^' in expression:
            base, exponent = expression.split('^', 1)
            base_val = float(base) if base.replace('.', '', 1).isdigit() else direct_calculator(base)
            exponent_val = float(exponent) if exponent.replace('.', '', 1).isdigit() else direct_calculator(exponent)
            return base_val ** exponent_val

        # Handle multiplication and division
        if '*' in expression:
            left, right = expression.split('*', 1)
            left_val = float(left) if left.replace('.', '', 1).isdigit() else direct_calculator(left)
            right_val = float(right) if right.replace('.', '', 1).isdigit() else direct_calculator(right)
            return left_val * right_val
        elif '/' in expression:
            left, right = expression.split('/', 1)
            left_val = float(left) if left.replace('.', '', 1).isdigit() else direct_calculator(left)
            right_val = float(right) if right.replace('.', '', 1).isdigit() else direct_calculator(right)
            if right_val == 0:
                raise ZeroDivisionError("Division by zero")
            return left_val / right_val

        # Handle addition and subtraction
        elif '+' in expression:
            left, right = expression.split('+', 1)
            left_val = float(left) if left.replace('.', '', 1).isdigit() else direct_calculator(left)
            right_val = float(right) if right.replace('.', '', 1).isdigit() else direct_calculator(right)
            return left_val + right_val
        elif '-' in expression and not expression.startswith('-'):
            left, right = expression.split('-', 1)
            left_val = float(left) if left.replace('.', '', 1).isdigit() else direct_calculator(left)
            right_val = float(right) if right.replace('.', '', 1).isdigit() else direct_calculator(right)
            return left_val - right_val
        elif expression.startswith('-'):
            return -direct_calculator(expression[1:])
        else:
            return float(expression)
    except Exception as e:
        print(f"Calculator error: {e}")
        return None

@tool
def ask_math_question(question: str) -> str:
    """Handles advanced math (exponentials, logs, absolutes, integrals, differentials)."""

    # Try direct calculation for basic arithmetic and exponentiation
    result = direct_calculator(question.strip())
    if result is not None:
        return f"The result of {question} is {result}"

    # Use sympy for advanced calculations
    try:
        # Convert Mathematica-like input to SymPy
        expr = parse_mathematica(question)
        result = sympy.simplify(expr)

        # Check if the result is a number or a symbolic expression
        if result.is_number:
            return f"The result of {question} is {float(result)}"  # Convert to float
        else:
            return f"The result of {question} is {result}" # Return symbolic result

    except (SyntaxError, TypeError, ValueError, sympy.SympifyError) as e:
        # Fallback to LLM if sympy fails
        try:
            print(f"SymPy failed: {e}, falling back to LLM")
            prompt = f"Solve the following math problem, showing steps if possible: {question}"
            model = ChatVertexAI(model="gemini-1.5-pro-002", temperature=0, max_tokens=1024, streaming=False)
            response = model.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as llm_e:
            return f"Unable to calculate: {question}. Error: {llm_e}"
    except Exception as e:
        return f"Unable to calculate: {question}. Error: {e}"


def handle_user_input(user_input, state=None):
    """Process user input, handling math directly."""
    if state is None:
        state = {"messages": []}

    new_message = HumanMessage(content=user_input)
    updated_state = {"messages": state["messages"] + [new_message]}

    # Check for math *after* adding the message
    if re.match(r'^[\d\s\+\-\*\/\(\)\.\^\|\w\[\]]+$', user_input.strip()):
        result = direct_calculator(user_input.strip())  # Try basic calculation first
        if result is not None:
            updated_state["messages"].append(AIMessage(content=f"The result of {user_input} is {result}"))

    return updated_state

@tool
def extract_text_from_document(file_path: str) -> str:
    """Extracts text from PDF/image."""
    try:
        text = ""
        if file_path.lower().endswith(".pdf"):
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text()
            doc.close()
        elif file_path.lower().endswith((".png", ".jpg", ".jpeg")):
            text = pytesseract.image_to_string(Image.open(file_path))
        else:
            raise ValueError(f"Unsupported file type: {os.path.splitext(file_path)[1]}.")
        text = re.sub(' +', ' ', text).strip()
        return f"File: {file_path} \n Extracted text: {text}"
    except Exception as e:
        return f"Error extracting text: {e}"

@tool
def detect_questions(text:str) -> list[str]:
    """Detects questions in text."""
    prompt = f"""Detect questions in:
    {text}
    Return a list.
    """
    try:
      response = llm.invoke([HumanMessage(content=prompt)])
      if response and response.content:  # Check if response and content exist
          return response.content.split("\n")
      else:
          return [] # Return empty list
    except Exception as e:
        print(f"Error in detect_questions: {e}")
        return []

@tool
def compare_answers(student_answers: str, answer_key: str) -> Dict:
    """Compares student answers to key."""
    results = {}
    def clean_answer(answer):
        answer = re.sub(r'[^\w\s]', '', answer).lower()
        return re.sub(' +', ' ', answer).strip()

    student_answers_list = [clean_answer(ans) for ans in re.split(r'\n|\. |\.',student_answers) if clean_answer(ans)]
    answer_key_list = [clean_answer(ans) for ans in re.split(r'\n|\. |\.',answer_key) if clean_answer(ans)]
    min_len = min(len(student_answers_list), len(answer_key_list))

    for i in range(min_len):
        student_answer = student_answers_list[i]
        correct_answer = answer_key_list[i]
        if student_answer == correct_answer:
            results[f"Question {i+1}"] = {"correct": True}
        elif student_answer in correct_answer or correct_answer in student_answer:
            results[f"Question {i+1}"] = {"correct": True, "feedback": "Partially Correct"}
        else:
            results[f"Question {i+1}"] = {"correct": False}

    for i in range(min_len, max(len(student_answers_list), len(answer_key_list))):
        results[f"Question {i+1}"] = {"correct": False, "feedback": "No answer"}
    return results

@tool
def grade_exam(comparison_results: Dict) -> str:
    """Grades exam based on comparison."""
    total_correct = sum(1 for res in comparison_results.values() if res["correct"])
    total_questions = len(comparison_results)
    score = (total_correct / total_questions) * 100
    feedback = f"Score: {score:.2f}% ({total_correct}/{total_questions} correct)\n"
    for question, result in comparison_results.items():
        feedback += f"{question}: {'Correct' if result['correct'] else 'Incorrect'}"
        if "feedback" in result:
            feedback += f" ({result['feedback']})"
        feedback += "\n"
    return feedback

@tool
def retrieve_docs(query: str) -> str:
    """Retrieves docs based on query."""
    retrieved_docs = retriever.invoke(query)
    ranked_docs = compressor.compress_documents(documents=retrieved_docs, query=query)
    return format_docs.format(docs=ranked_docs)

@tool
def should_continue() -> None:
    """Use if enough context to respond."""
    return None

tools = [
    retrieve_docs,
    should_continue,
    extract_text_from_document,
    compare_answers,
    grade_exam,
    detect_questions,
    ask_math_question
]

llm = ChatVertexAI(model=LLM, temperature=0, max_tokens=1024, streaming=True)
inspect_conversation = inspect_conversation_template | llm.bind_tools(tools=tools)
response_chain = rag_template | llm

def inspect_conversation_node(state: MessagesState, config: RunnableConfig):
    """Inspects conversation, handling math or delegating."""
    messages = state.get("messages", [])
    if not messages:
        return {"messages": [HumanMessage(content="")]}

    last_message = messages[-1]
    if isinstance(last_message, HumanMessage):
        if isinstance(last_message.content, list):
            content_str = " ".join(str(item) for item in last_message.content)
        else:
            content_str = str(last_message.content)
        # More robust regex
        if re.match(r'^[\d\s\+\-\*\/\(\)\.\^\|\w\[\]]+$', content_str.strip()):
            result = ask_math_question(content_str.strip())
            return {"messages": messages + [AIMessage(content=result)]}

    try:
        response = inspect_conversation.invoke({"messages": messages}, config)
        return {"messages": response}
    except Exception as e:
        return {"messages": [AIMessage(content="Error. Try rephrasing.")]}

def generate_node(state: MessagesState, config: RunnableConfig) -> dict[str, BaseMessage]:
    """Generates response using RAG."""
    response = response_chain.invoke(state, config)
    return {"messages": [response]}

workflow = StateGraph(MessagesState)
workflow.add_node("agent", inspect_conversation_node)
workflow.add_node("generate", generate_node)
workflow.set_entry_point("agent")
workflow.add_node("tools", ToolNode(tools=tools, handle_tool_errors=False))
workflow.add_edge("agent", "tools")
workflow.add_edge("tools", "generate")
workflow.add_edge("generate", END)
agent = workflow.compile()


def process_exam(file_path: str):
    """Processes the exam, extracts text, and answers questions."""
    try:
        extracted_text = extract_text_from_document(file_path)
        if extracted_text.startswith("Error"):
            return extracted_text

        questions = detect_questions(extracted_text)
        if isinstance(questions, str):
            return f"Error detecting questions: {questions}"
        if not questions:
            return "No questions found in the document."

        all_answers = []
        for question in questions:
            print(f"Answering question: {question}")
            response = agent.invoke({"messages": [HumanMessage(content=question)]})
            if 'messages' in response and response['messages']:
                if isinstance(response['messages'][-1], AIMessage):
                    answer = response['messages'][-1].content
                elif isinstance(response['messages'][-1], HumanMessage):
                     answer = response['messages'][-1].content
                else:
                    answer = str(response['messages'][-1])
            else:
                answer = "No answer found."
            all_answers.append(f"Question: {question}\nAnswer: {answer}\n")

        return "\n".join(all_answers)

    except Exception as e:
        return f"Error processing exam: {e}"

if __name__ == "__main__":
    exam_results = process_exam("Practice Exam2-truncated.pdf")  # Correct path
    print(exam_results)