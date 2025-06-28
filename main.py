import streamlit as st
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_groq import ChatGroq

# Set up Streamlit page
st.set_page_config(page_title="Writing Text Summarization")
st.title("Writing Text Summarization")

# --- Helper Functions ---

def load_LLM(groq_api_key: str, model_name: str = "llama3-70b-8192"):
    if not groq_api_key:
        st.warning("Please provide a valid Groq API key.")
        return None
    return ChatGroq(groq_api_key=groq_api_key, model_name=model_name)

def generate_response(txt, groq_api_key):
    llm = load_LLM(groq_api_key)
    if not llm:
        return "Invalid or missing API key."

    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(txt)
    docs = [Document(page_content=t) for t in texts]

    chain = load_summarize_chain(llm, chain_type="map_reduce")
    return chain.run(docs)

# --- User Inputs ---

txt_input = st.text_area("Enter your text", "", height=200)

with st.form("summarize_form", clear_on_submit=True):
    groq_api_key = st.text_input(
        label="Groq API Key",
        placeholder="Ex: gsk-live-xxxxxxxxxxxxxxxxx",
        key="groq_api_key_input",
        type="password"
    )
    submitted = st.form_submit_button("Submit")

    if submitted:
        if txt_input.strip() == "":
            st.warning("Please enter some text.")
        elif not groq_api_key.startswith("gsk"):
            st.warning("Please enter a valid Groq API Key.")
        else:
            response = generate_response(txt_input, groq_api_key)
            st.info(response)
