import os
from dotenv import load_dotenv
from datetime import datetime

from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

from task2.main import chain_3

load_dotenv()

# ── LLM ──────────────────────────────────────────────────────────────────────
llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0
)

# ── EMBEDDINGS ────────────────────────────────────────────────────────────────
embeddings = AzureOpenAIEmbeddings(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

# ── VECTOR STORE (reusing aiIntro.txt from task3) ────────────────────────────
file_path = os.path.join(os.path.dirname(__file__), "..", "task3", "aiIntro.txt")
loader = TextLoader(file_path, encoding="utf-8")
docs = loader.load()

splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.split_documents(docs)

vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()

# ── TOOL 1: RETRIEVER ─────────────────────────────────────────────────────────
@tool
def retrieve_text(query: str) -> str:
    """Retrieve relevant text from the AI document based on a query."""
    retrieved = retriever.invoke(query)
    return "\n".join([d.page_content for d in retrieved])


# ── TOOL 2: SUMMARIZER ────────────────────────────────────────────────────────
@tool
def summarize_text(text: str) -> str:
    """Summarize the given text into exactly 3 sentences."""
    return chain_3.invoke({"text": text})


# ── TOOL 3: WORD COUNTER ──────────────────────────────────────────────────────
@tool
def count_words(text: str) -> str:
    """Count the number of words in the given text."""
    count = len(text.split())
    return f"Word count: {count}"


# ── TOOL 4: GET CURRENT DATE ──────────────────────────────────────────────────
@tool
def get_current_date(query: str = "") -> str:
    """Returns today's current date."""
    return f"Today's date is: {datetime.now().strftime('%B %d, %Y')}"


# ── TOOL 5: MOCK WEB SEARCH ───────────────────────────────────────────────────
@tool
def mock_web_search(query: str) -> str:
    """Performs a mock web search and returns a static response about recent AI trends."""
    return (
        "Recent AI updates: Large language models like GPT and Claude are advancing rapidly. "
        "AI regulation discussions are ongoing globally. "
        "Multimodal AI systems combining text, image, and audio are becoming mainstream. "
        "AI in healthcare is seeing major investments, especially in diagnostics and drug discovery."
    )


# ── AGENT ─────────────────────────────────────────────────────────────────────
from langchain.agents import create_agent

tools = [retrieve_text, summarize_text, count_words, get_current_date, mock_web_search]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "You are a helpful assistant with access to tools. "
        "Use the appropriate tools to answer the user's request step by step. "
        "Always use tools when needed rather than relying on your own knowledge."
    )
)

# ── TEST 1: Summarize + Date ──────────────────────────────────────────────────
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

# ── TEST 2: Summarize + Mock Web Search ──────────────────────────────────────
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