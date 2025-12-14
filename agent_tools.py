# agent_tools.py - Minimal Real Agent

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain.tools import tool
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages
from rag_utils import search_documents, get_openai_client
import os

# tools definitions
@tool
def search_tool(query: str) -> str:
    """Search SOW documents"""
    results = search_documents(query, top_k=3)
    if not results:
        return "No documents found"
    return "\n\n".join([f"{doc['title']}: {doc['chunk'][:300]}" for doc in results])


@tool
def risk_check_tool(content: str) -> str:
    """Check for risks in SOW document content. 
    Only identify risks present in the content.
    Do not make up risks. 
    Return 'Document not found' if no relevant document is found."""
    
    risk_analysis_prompt = f"""
    You are a contract risk analyst. Analyze this Statement of Work content for risks:

    {content}

    Identify ALL risks in these categories:
    1. **Financial Risks**: Payment terms, pricing, budget issues
    2. **Legal Risks**: Missing clauses (warranty, IP, liability, termination)
    3. **Timeline Risks**: Unrealistic schedules, unclear milestones
    4. **Scope Risks**: Vague deliverables, scope creep indicators
    5. **Operational Risks**: Resource availability, dependencies
    6. **Compliance Risks**: Regulatory requirements, data protection

    For each risk found:
    - State the risk clearly
    - Explain the impact
    - Provide a recommendation

    Format as:
    **Risk Category**
    ⚠️ Risk: [description]
    Impact: [what could go wrong]
    Recommendation: [how to mitigate]

    If no significant risks, state "No major risks detected"
    """

    client = get_openai_client()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert contract risk analyst. Be thorough but precise."},
            {"role": "user", "content": risk_analysis_prompt}
        ],
        temperature=0.3  # Lower temperature for more consistent analysis
    )
    
    return response.choices[0].message.content


class AgentState(TypedDict):
    messages: Annotated[List, add_messages]


# Create LLM with tools
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ACCOUNT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2025-01-01-preview",
    deployment_name="gpt-4o",
)

tools = [search_tool, risk_check_tool]
llm_with_tools = llm.bind_tools(tools)


def agent_node(state: AgentState):
    """Agent decides: use tool or answer?"""
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def should_continue(state: AgentState):
    """If agent called a tool, execute it. Otherwise, done."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))

workflow.set_entry_point("agent")

# conditional routing
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})

workflow.add_edge("tools", "agent")

agent = workflow.compile()

def analyze(question: str) -> str:
    """Ask the agent anything"""
    result = agent.invoke({"messages": [HumanMessage(content=question)]})
    return result["messages"][-1].content
