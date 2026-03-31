import os
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent

from task2.main import chain_3
from task3.main import retriever

load_dotenv()

# ── LLM ──────────────────────────────────────────────────────────────────────
llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0
)

# ── TOOL 1: RETRIEVER ─────────────────────────────────────────────────────────
@tool
def retrieve_text(query: str) -> str:
    """Retrieve relevant text from the AI document based on a query."""
    docs = retriever.invoke(query)
    return "\n".join([d.page_content for d in docs])

# ── TOOL 2: SUMMARIZER ────────────────────────────────────────────────────────
@tool
def summarize_text(text: str) -> str:
    """Summarizes given text into exactly 3 sentences."""
    return chain_3.invoke({"text": text})

# ── TOOL 3: WORD COUNTER ──────────────────────────────────────────────────────
@tool
def count_words(text: str) -> str:
    """Counts the number of words in the given text."""
    count = len(text.split())
    return f"Word count: {count}"

# ── AGENT ─────────────────────────────────────────────────────────────────────
tools = [retrieve_text, summarize_text, count_words]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "You are a helpful assistant. "
        "When asked to find and summarize text, first use retrieve_text to find "
        "relevant content, then use summarize_text to summarize it. "
        "When asked for word count, use count_words on the summary."
    )
)

if __name__ == "__main__":
    # ── TEST 1: Find and Summarize ────────────────────────────────────────────
    print("\n=== TEST 1: Find and Summarize AI Breakthroughs ===")
    result1 = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": "Find and summarize text about AI breakthroughs from the document."
            }
        ]
    })
    print(result1["messages"][-1].content)

    # ── TEST 2: Find, Summarize + Word Count ──────────────────────────────────
    print("\n=== TEST 2: Find, Summarize + Word Count ===")
    result2 = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": "Find and summarize text about AI breakthroughs from the document. Also give me the word count of the summary."
            }
        ]
    })
    print(result2["messages"][-1].content)