import os
from dotenv import load_dotenv
from datetime import datetime

from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent

from task5.main import retrieve_text, summarize_text, count_words

load_dotenv()

# LLM
llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0
)

# TOOL 4: GET CURRENT DATE
@tool
def get_current_date(query: str = "") -> str:
    """Returns today's current date."""
    return f"Today's date is: {datetime.now().strftime('%B %d, %Y')}"

# TOOL 5: MOCK WEB SEARCH
@tool
def mock_web_search(query: str) -> str:
    """Performs a mock web search and returns a static response about recent AI trends."""
    return (
        "Recent AI updates: Large language models like GPT and Claude are advancing rapidly. "
        "AI regulation discussions are ongoing globally. "
        "Multimodal AI systems combining text, image, and audio are becoming mainstream. "
        "AI in healthcare is seeing major investments, especially in diagnostics and drug discovery."
    )

# (Task 5 tools + 2 new tools)
tools = [retrieve_text, summarize_text, count_words, get_current_date, mock_web_search]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "You are a helpful assistant. "
        "Use retrieve_text to find content from the document, "
        "summarize_text to summarize any text, "
        "count_words for word counts, "
        "get_current_date for today's date, "
        "and mock_web_search for recent updates."
    )
)

# TEST 1: Summarize + Date
text1 = """
Artificial intelligence is reshaping industries by automating complex tasks,
improving decision-making, and enabling new capabilities in healthcare, finance,
and education. AI systems now assist doctors in diagnosing diseases, help banks
detect fraud in real time, and personalize learning experiences for students.
Despite these advances, concerns about bias, transparency, and job displacement
remain critical challenges that researchers and policymakers must address.
"""

print("\n=== TEST 1: Summarize + Today's Date ===")
result1 = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": f"Summarize this text about AI and tell me today's date:\n{text1}"
        }
    ]
})
print(result1["messages"][-1].content)

# TEST 2: Summarize AI Trends + Mock Web Search
print("\n=== TEST 2: Summarize AI Trends + Mock Web Search ===")
result2 = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Summarize AI trends from the document and search for recent updates."
        }
    ]
})
print(result2["messages"][-1].content)