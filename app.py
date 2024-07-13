import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader
import os

# Initialize the OpenAI client using the secret from secrets.toml
API_KEY_ = st.secrets["API_KEY_"]
openai = OpenAI(
    api_key=API_KEY_,
    base_url="https://api.deepinfra.com/v1/openai"
)

# Initialize the OpenAI client
# API_KEY_ = os.environ.get("API_KEY_")
# openai = OpenAI(
#     api_key=API_KEY_,
#     base_url="https://api.deepinfra.com/v1/openai"
# ) 

def get_openai_response(message_history):
    """
    Get a response from the OpenAI model.

    Args:
        message_history (list): List of messages in the conversation history.

    Returns:
        str: The response from the OpenAI model.
    """
    chat_completion = openai.chat.completions.create(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        messages=message_history
    )
    return chat_completion.choices[0].message.content

def send_message():
    """
    Send the user's message to the OpenAI model and get the response.
    Updates the session state with the new messages.
    """
    user_message = {"role": "user", "content": st.session_state.user_input}
    st.session_state.history.append(user_message)

    # Get response from OpenAI
    openai_response = get_openai_response(st.session_state.history)
    st.session_state.history.append({"role": "assistant", "content": openai_response})
    st.session_state.user_input = ""

def extract_text_from_pdf(pdf_file):
    """
    Extract text from an uploaded PDF file.

    Args:
        pdf_file (UploadedFile): The uploaded PDF file.

    Returns:
        str: The extracted text from the PDF file.
    """
    reader = PdfReader(pdf_file)
    text = ""
    for page in range(len(reader.pages)):
        text += reader.pages[page].extract_text()
    return text

def get_rag_response(user_query, pdf_text):
    """
    Get a response from the OpenAI model using the user's query and context from a PDF.

    Args:
        user_query (str): The user's query.
        pdf_text (str): The extracted text from the PDF.

    Returns:
        str: The response from the OpenAI model.
    """
    augmented_input = f"{user_query}\n\nContext from PDF:\n{pdf_text[:1000]}"  # Limiting context length
    message_history = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": augmented_input}
    ]
    return get_openai_response(message_history)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'user_input' not in st.session_state:
    st.session_state['user_input'] = ""
if 'pdf_text' not in st.session_state:
    st.session_state['pdf_text'] = ""

# Sidebar for PDF upload
st.sidebar.header("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    st.session_state['pdf_text'] = extract_text_from_pdf(uploaded_file)
    st.session_state.history.append({"role": "system", "content": "PDF content uploaded and extracted."})

# Display conversation history
chat_container = st.container()

for message in st.session_state.history:
    if message['role'] == "user":
        chat_container.markdown(f"""
        <div style='text-align: right; padding: 10px; border-radius: 10px; margin: 5px; display: flex; align-items: center;'>
            <span style='flex: 1;'></span>
            <span style='padding-right: 10px;'>{message['content']}</span>
            <img src='https://img.icons8.com/color/48/000000/user-male-circle.png' width='24' style='vertical-align: middle;'> 
        </div>
        """, unsafe_allow_html=True)
    else:
        chat_container.markdown(f"""
        <div style='text-align: left; padding: 10px; border-radius: 10px; margin: 5px; display: flex; align-items: center;'>
            <img src='https://img.icons8.com/color/48/000000/robot-2.png' width='24' style='vertical-align: middle;'> 
            <span style='padding-left: 10px;'>{message['content']}</span>
        </div>
        """, unsafe_allow_html=True)

# Input box and Send button at the bottom
st.text_input("", key="user_input", on_change=send_message, value="", placeholder="Ask something...")

# Custom CSS for styling
st.markdown("""
    <style>
        .text-center {
            text-align: center;
        }
        .mt-5 {
            margin-top: 3rem;
        }
        .mb-4 {
            margin-bottom: 1.5rem;
        }
    </style>
    """, unsafe_allow_html=True)

# Example usage of RAG
if st.session_state.user_input:
    user_query = st.session_state.user_input
    pdf_context = st.session_state.pdf_text
    rag_response = get_rag_response(user_query, pdf_context)
    st.session_state.history.append({"role": "assistant", "content": rag_response})
    st.session_state.user_input = ""
