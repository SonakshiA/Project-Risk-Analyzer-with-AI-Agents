# RAG-Powered Statement of Work (SOW) Analysis System

An intelligent document analysis system that uses Retrieval Augmented Generation (RAG) and AI Agents to analyze Statement of Work documents. Built with Azure AI Search, Azure OpenAI, and LangGraph.

## 🎯 Features

- **Hybrid Search**: Combines semantic search, vector search, and full-text search for optimal document retrieval
- **AI Agent**: Autonomous agent that can search documents and perform risk analysis
- **Risk Detection**: Automated identification of contract risks and red flags
- **Interactive Chat**: Streamlit-based UI for natural language queries
- **Document Chunking**: Intelligent text splitting with overlap for better context preservation
- **Azure Integration**: Full Azure cloud stack (AI Search, OpenAI, Blob Storage)

## 🏗️ Architecture

```
┌─────────────┐
│   User UI   │ (Streamlit)
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│    AI Agent         │ (LangGraph)
│  ┌──────────────┐   │
│  │ Agent Node   │   │ ← Decides what to do
│  └──────┬───────┘   │
│         │           │
│  ┌──────▼───────┐   │
│  │ Tool Node    │   │ ← Executes tools
│  │ - Search     │   │
│  │ - Risk Check │   │
│  └──────────────┘   │
└──────────┬──────────┘
           │
           ▼
┌──────────────────────┐
│  Azure AI Search     │
│  ┌────────────────┐  │
│  │ Index          │  │ ← Stores chunks + embeddings
│  │ Skillset       │  │ ← Chunks + embeds documents
│  │ Indexer        │  │ ← Orchestrates pipeline
│  └────────────────┘  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Azure Blob Storage  │ ← Source documents (PDFs, TXT)
└──────────────────────┘
           │
           ▼
┌──────────────────────┐
│  Azure OpenAI        │
│  - Embeddings (3072) │
│  - GPT-4o            │
└──────────────────────┘
```

## 📂 Project Structure

```
.
├── rag.ipynb                  # Setup notebook (index, skillset, indexer)
├── rag.py                     # Setup script (alternative to notebook)
├── rag_utils.py               # Core RAG functions (search, generate answer)
├── agent_tools.py             # AI Agent with LangGraph
├── test.py                    # Streamlit UI
├── main.py                    # Entry point
├── sample_sows/               # Sample Statement of Work documents
│   ├── contoso_web_development.txt
│   ├── acme_data_migration.txt
│   └── northwind_mobile_app.txt
├── .env                       # Environment variables (keys)
└── README.md                  # This file
```

## 🚀 Setup

### Prerequisites

- Python 3.10+
- Azure subscription with:
  - Azure AI Search service
  - Azure OpenAI service
  - Azure Blob Storage account
  - Azure Cognitive Services (optional)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/SonakshiA/RAG-with-Ollama.git
   cd RAG-with-Ollama
   ```

2. **Install dependencies**
   ```bash
   pip install python-dotenv azure-search-documents azure-identity openai streamlit langgraph langchain langchain-openai
   ```

3. **Configure environment variables**
   
   Create a `.env` file in the root directory:
   ```env
   AZURE_SEARCH_SERVICE=https://your-search-service.search.windows.net
   AZURE_SEARCH_KEY=your_search_key
   AZURE_OPENAI_ACCOUNT=https://your-openai.openai.azure.com
   AZURE_OPENAI_KEY=your_openai_key
   AZURE_STORAGE_CONNECTION=your_blob_connection_string
   ```

4. **Upload documents to Azure Blob Storage**
   
   Upload your SOW documents to a container named `sow-container` in your Azure Blob Storage account.

5. **Run the setup (one-time)**
   
   Option A - Using Jupyter Notebook:
   ```bash
   jupyter notebook rag.ipynb
   # Run all cells to create index, skillset, and indexer
   ```
   
   Option B - Using Python script:
   ```bash
   python rag.py
   ```

6. **Launch the Streamlit app**
   ```bash
   streamlit run test.py
   ```

## 💡 Usage

### Simple RAG Query
```python
from rag_utils import generate_answer

answer = generate_answer("What are the deliverables in the Contoso project?")
print(answer)
```

### Using the AI Agent
```python
from agent_tools import analyze

# Agent will automatically search and analyze
answer = analyze("What are the risks in the Acme project?")
print(answer)
```

### Streamlit UI
1. Run `streamlit run test.py`
2. Type your question in the text input
3. View the AI-generated answer with source citations

## 🔧 Configuration

### Index Schema
- **id**: Document identifier
- **title**: Document title
- **chunk_id**: Unique chunk identifier (key field)
- **chunk**: Text content
- **text_vector**: 3072-dimension embeddings (text-embedding-3-large)

### Skillset Pipeline
1. **SplitSkill**: Chunks documents (2000 chars, 500 overlap)
2. **AzureOpenAIEmbeddingSkill**: Generates embeddings for each chunk

### Search Configuration
- **Vector Search**: HNSW algorithm with Azure OpenAI vectorizer
- **Semantic Search**: Microsoft's semantic ranking
- **Hybrid Search**: Combines vector + semantic + keyword search

## 🤖 AI Agent

The agent uses LangGraph to orchestrate multi-step reasoning:

**Tools Available:**
- `search_tool`: Searches SOW documents
- `risk_check_tool`: Identifies contract risks

**Agent Workflow:**
1. User asks question
2. Agent decides if it needs to search
3. If yes: calls search_tool
4. Analyzes results
5. If needed: calls risk_check_tool
6. Provides final answer

**Example Agent Execution:**
```
User: "What are the risks in Contoso?"

Agent: "I'll search for Contoso first"
→ search_tool("Contoso")
→ Finds: [Payment terms, timeline, deliverables]

Agent: "Now I'll check for risks"
→ risk_check_tool(results)
→ Finds: "No warranty, Full payment after completion"

Agent: "Here's my analysis..."
→ Returns comprehensive risk report
```

## 📊 Sample Documents

Three sample SOW documents are included in `sample_sows/`:

1. **Contoso Web Development** - $250K, 6 months, e-commerce platform
2. **Acme Data Migration** - $480K, 4 months, cloud migration
3. **Northwind Mobile App** - $320K, 5 months, iOS/Android app

## 🔍 Key Components

### `rag_utils.py`
- `get_search_client()`: Azure Search client
- `get_openai_client()`: Azure OpenAI client
- `search_documents()`: Hybrid search with vector + semantic
- `generate_answer()`: RAG query with grounded prompts

### `agent_tools.py`
- `search_tool()`: Document search tool
- `risk_check_tool()`: Risk detection tool
- `analyze()`: Main agent interface

### `test.py`
- Streamlit UI with chat interface
- Spinner for loading states
- Answer display with formatting

## 🎓 Learning Resources

**Concepts Used:**
- Retrieval Augmented Generation (RAG)
- Vector embeddings and similarity search
- Semantic search and reranking
- AI agents with tool calling
- LangGraph for agent orchestration
- Azure AI services integration

**Documentation:**
- [Azure AI Search](https://learn.microsoft.com/en-us/azure/search/)
- [Azure OpenAI](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [LangChain](https://python.langchain.com/)

## 🐛 Troubleshooting

**Import errors?**
```bash
pip install python-dotenv azure-search-documents azure-identity openai streamlit langgraph langchain-openai
```

**No documents found?**
- Check if indexer ran successfully: See `rag.ipynb` indexer status cells
- Verify documents are in blob storage container `sow-container`
- Check Azure Search index has documents (use Azure Portal)

**Agent returns None?**
- Ensure all tools have docstrings
- Check Azure OpenAI API key is valid
- Verify `tool_calls` attribute exists in messages

**Authentication errors?**
- Use `AzureKeyCredential` instead of `DefaultAzureCredential`
- Verify API keys in `.env` file

## 📝 License

This project is for educational purposes as part of the AI Apprentice Capstone project.

## 👥 Contributors

- Sonakshi Arora ([@SonakshiA](https://github.com/SonakshiA))

## 🙏 Acknowledgments

- Azure AI Search for powerful hybrid search capabilities
- LangChain/LangGraph for agent framework
- OpenAI for embedding and language models
