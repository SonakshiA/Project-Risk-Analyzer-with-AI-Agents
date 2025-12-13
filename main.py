import streamlit as st
from rag_utils import generate_answer
from agent_tools import analyze

st.title("Talk to your Doc")

mode = st.radio("Select Mode: ", ["Simple RAG", "Contract Agent"])

question = st.text_input("Ask a question about the document:")
if question:
    with st.spinner("Generating answer..."):
        if mode == "Simple RAG":
            answer = generate_answer(question)
        else:
            answer = analyze(question)
    st.write("Answer:", answer)