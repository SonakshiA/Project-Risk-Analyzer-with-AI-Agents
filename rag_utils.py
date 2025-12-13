"""
RAG Utility Functions for Azure AI Search + OpenAI
Modularized to avoid recreating index/skillset on every query
"""

import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
from azure.search.documents import SearchClient # client to interact with the Search Index
from azure.search.documents.models import VectorizableTextQuery

load_dotenv()

AZURE_SEARCH_SERVICE: str = os.getenv("AZURE_SEARCH_SERVICE")
AZURE_SEARCH_KEY: str = os.getenv("AZURE_SEARCH_KEY")
AZURE_OPENAI_ACCOUNT: str = os.getenv("AZURE_OPENAI_ACCOUNT")
AZURE_OPENAI_KEY: str = os.getenv("AZURE_OPENAI_KEY")
AZURE_AI_MULTISERVICE_ACCOUNT: str = os.getenv("AZURE_AI_MULTISERVICE_ACCOUNT")
AZURE_AI_MULTISERVICE_KEY: str = os.getenv("AZURE_AI_MULTISERVICE_KEY")
AZURE_STORAGE_CONNECTION: str = os.getenv("AZURE_STORAGE_CONNECTION")
INDEX_NAME = "sow-index"
DEPLOYMENT_NAME = "gpt-4o"
GROUNDED_PROMPT = """
You are an AI assistant that helps users learn from the information found in the source material.
Answer the query using only the sources provided below.
Use bullets if the answer has multiple points.
If the answer is longer than 3 sentences, provide a summary.
Answer ONLY with the facts listed in the list of sources below. Cite your source when you answer the question
If there isn't enough information below, say you don't know.
Do not generate answers that don't use the sources below.
Query: {query}
Sources:\n{sources}
"""


def get_search_client():
    return SearchClient(
    endpoint=AZURE_SEARCH_SERVICE,
    index_name=INDEX_NAME,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY)
)
    
def get_openai_client():
    return AzureOpenAI(
    api_version="2025-01-01-preview",
    azure_endpoint=AZURE_OPENAI_ACCOUNT,
    api_key=AZURE_OPENAI_KEY
)

def search_documents(query: str, top_k: int =5):
    search_client = get_search_client()

    vectorized_query = VectorizableTextQuery(text=query, k_nearest_neighbors=top_k, fields="text_vector")

    results = search_client.search(
        query_type="semantic",
        semantic_configuration_name="my-semantic-config",
        search_text=query, 
        vector_queries=[vectorized_query], 
        select=["title", "chunk"],  #which fields to return
        top=top_k)
    
    return results

def generate_answer(query: str, top_k:int=5) -> str:
    openai_client = get_openai_client()

    results = search_documents(query, top_k)

    response = openai_client.chat.completions.create(
    model=DEPLOYMENT_NAME,
    messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that helps people find information."
            },
            {
                "role": "user",
                "content": GROUNDED_PROMPT.format(
                    query=query,
                    sources="\n".join([f"- {doc['chunk']} (Source: {doc['title']})" for doc in results])
                )
            }
        ]
    )

    return response.choices[0].message.content



