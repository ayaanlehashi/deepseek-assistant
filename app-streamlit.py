import streamlit as st
import requests
import os
import logging
import PyPDF2
from typing import List, Dict


API_URL = "https://api.deepseek.com/v1/chat/completions"
LOG_FILENAME = "deepseek_dashboard.log"
MAX_CONTEXT_MESSAGES = 8  
MAX_FILE_CONTENT = 1000   

def configure_logging():
    """Configures logging system with file and console handlers"""
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(LOG_FILENAME)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler]
    )

configure_logging()
logger = logging.getLogger("DeepSeekDashboard")


class ChatMemory:
    """Manages conversation history with token-aware trimming"""

    def __init__(self, max_messages: int = MAX_CONTEXT_MESSAGES):
        self.max_messages = max_messages
        self.initialize_session()

    def initialize_session(self):
        """Initialize session state variables"""
        if "messages" not in st.session_state:
            st.session_state["messages"] = []
            st.session_state["full_history"] = []

    def add_message(self, role: str, content: str):
        """Adds message with automatic history trimming"""
        st.session_state.messages.append({"role": role, "content": content})
        st.session_state.full_history.append({"role": role, "content": content})

        while len(st.session_state.full_history) > self.max_messages:
            removed = st.session_state.full_history.pop(0)
            logger.debug(f"Trimming message: {removed['content'][:50]}...")

    def get_context(self, system_prompt: str) -> List[Dict]:
        """Returns optimized conversation context"""
        return [{"role": "system", "content": system_prompt}] + st.session_state.full_history[-self.max_messages:]

    def clear_memory(self):
        """Resets conversation history"""
        st.session_state.messages = []
        st.session_state.full_history = []


def query_deepseek(
    prompt: str,
    system_prompt: str,
    memory: ChatMemory,
    model: str = "deepseek-chat",
    temperature: float = 0.7
) -> Dict:
    """Handles DeepSeek API communication with cost optimization"""
    headers = {
        "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}",
        "Content-Type": "application/json",
    }

    try:
        payload = {
            "model": model,
            "messages": memory.get_context(system_prompt) + [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }

        context_count = len(payload["messages"])
        logger.info(f"Context optimization: Sending {context_count} messages (reduced from {len(st.session_state.full_history) + 2})")

        with st.spinner("Processing your request..."):
            response = requests.post(API_URL, headers=headers, json=payload)
            response.raise_for_status()

            if response.status_code == 200:
                response_data = response.json()
                assistant_response = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")

                memory.add_message("user", prompt)
                memory.add_message("assistant", assistant_response)

                return response_data

        logger.error(f"API returned non-200 status: {response.status_code}")
        return None

    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {str(e)}", exc_info=True)
        st.error(f"API Error: {str(e)}")
        return None


def process_uploaded_files(files) -> str:
    """Processes uploaded files with aggressive truncation"""
    processed_content = []

    for file in files:
        try:
            if file.type == "application/pdf":
                pdf_reader = PyPDF2.PdfReader(file)
                text = " ".join(page.extract_text() or "" for page in pdf_reader.pages)
                processed_content.append(f"PDF_CONTENT:{file.name}: {text[:MAX_FILE_CONTENT]}...")
            else:
                content = file.read().decode("utf-8", errors="replace")
                processed_content.append(f"FILE_CONTENT:{file.name}: {content[:MAX_FILE_CONTENT]}...")
        except Exception as e:
            logger.error(f"Error processing {file.name}: {str(e)}")
            st.warning(f"Error processing {file.name}: {str(e)}")

    return "\n".join(processed_content)


def main_interface():
    """Main Streamlit interface with cost warnings"""
    st.set_page_config(page_title="DeepSeek Pro", layout="wide", page_icon="🧠")
    st.title("DeepSeek AI Assistant")



    # Sidebar controls
    with st.sidebar:
        st.title("Control Panel")
        model_choice = st.selectbox("AI Model", ["deepseek-chat", "deepseek-reasoner"], index=0)
        temperature = st.slider("Creativity Level", 0.0, 1.0, 0.7, 0.05)
        system_prompt = st.text_area(
            "System Role",
            value="You are an expert AI assistant. Provide detailed, accurate responses.",
            height=150
        )
        uploaded_files = st.file_uploader(
            "Add Knowledge Files",
            accept_multiple_files=True,
            type=None
        )

    # Initialize memory system
    memory = ChatMemory()


    # Display chat history
    for msg in st.session_state.get("messages", []):
        role = "user" if msg["role"] == "user" else "assistant"
        with st.chat_message(role):
            content = msg["content"]
            if "FILE_CONTENT" in content or "PDF_CONTENT" in content:
                parts = content.split(":", 2)
                st.write(parts[0])
                with st.expander("Attached Files"):
                    st.text(parts[-1])
            else:
                st.write(content)

    # Handle user input
    if user_input := st.chat_input("Ask anything..."):
        file_context = process_uploaded_files(uploaded_files) if uploaded_files else ""
        full_prompt = f"{user_input}\n{file_context}" if file_context else user_input

        with st.chat_message("user"):
            st.write(user_input)
            if file_context:
                with st.expander("Attached Files"):
                    st.text(file_context)

        with st.chat_message("assistant"):
            try:
                response = query_deepseek(
                    prompt=full_prompt,
                    system_prompt=system_prompt,
                    memory=memory,
                    model=model_choice,
                    temperature=temperature
                )

                if response and (content := response.get("choices", [{}])[0].get("message", {}).get("content", "")):
                    st.markdown(content)
                else:
                    st.error("Failed to get valid response from API")

            except Exception as e:
                st.error(f"Communication error: {str(e)}")
                logger.exception("Unexpected error in main interface")


if __name__ == "__main__":
    if not os.getenv("DEEPSEEK_API_KEY"):
        st.error("Missing API key! Set DEEPSEEK_API_KEY environment variable.")
    else:
        main_interface()