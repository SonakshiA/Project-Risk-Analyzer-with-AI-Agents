import streamlit as st
from rag_utils import generate_answer
from agent_tools import analyze

st.title("Talk to your Doc💡")
st.write("An intelligent document analysis system that uses RAG and AI Agents to analyze **Statement of Work** documents. Built with Azure AI Search, Azure OpenAI, and LangGraph.")

st.write("""Prompt Suggestions 🧠:\n 
         1. Tell me something about the Contoso project deliverables?\n
         2. Tell me the risks in Northwind Traders project?""")

mode = st.radio("Select Mode: ", ["Simple RAG", "Contract Agent (For Project Risk Assessment)"])

question = st.text_input("Ask a question about the document:")
if question:
    with st.spinner("Generating answer..."):
        if mode == "Simple RAG":
            answer = generate_answer(question)
        else:
            answer = analyze(question)
    st.write("Answer:", answer)