import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings

from task2.main import chain_3

load_dotenv()

# ── EMBEDDINGS ──────────────────────────────────────────────────────────────
embeddings = AzureOpenAIEmbeddings(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

# ── SPLITTER (150 chars, 30 overlap as per task) ─────────────────────────────
splitter = CharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=30,
    separator="\n"
)

# ── 1. LOAD PDF ──────────────────────────────────────────────────────────────
pdf_path = os.path.join(os.path.dirname(__file__), "ai_ethics.pdf")
pdf_loader = PyPDFLoader(pdf_path)
pdf_docs = pdf_loader.load()

pdf_chunks = splitter.split_documents(pdf_docs)
print(f"PDF chunks: {len(pdf_chunks)}")

pdf_vectorstore = FAISS.from_documents(pdf_chunks, embeddings)
pdf_retriever = pdf_vectorstore.as_retriever()

# ── 2. LOAD WEBPAGE ──────────────────────────────────────────────────────────
url = "https://www.ibm.com/think/topics/artificial-intelligence-trends"
web_loader = WebBaseLoader(url)
web_docs = web_loader.load()

web_chunks = splitter.split_documents(web_docs)
print(f"Web chunks: {len(web_chunks)}")

web_vectorstore = FAISS.from_documents(web_chunks, embeddings)
web_retriever = web_vectorstore.as_retriever()

# ── 3. QUERY BOTH WITH "AI challenges" ───────────────────────────────────────
query = "AI challenges"

print("\n=== PDF: Retrieved Chunks ===")
pdf_retrieved = pdf_retriever.invoke(query)
pdf_text = "\n".join([doc.page_content for doc in pdf_retrieved])
print(pdf_text[:500], "...")

print("\n=== WEB: Retrieved Chunks ===")
web_retrieved = web_retriever.invoke(query)
web_text = "\n".join([doc.page_content for doc in web_retrieved])
print(web_text[:500], "...")

# ── 4. SUMMARIZE EACH ────────────────────────────────────────────────────────
print("\n=== PDF SUMMARY ===")
pdf_summary = chain_3.invoke({"text": pdf_text})
print(pdf_summary)

print("\n=== WEB SUMMARY ===")
web_summary = chain_3.invoke({"text": web_text})
print(web_summary)

# ── 5. COMPARISON NOTE ───────────────────────────────────────────────────────
print("\n=== COMPARISON ===")
print("PDF Summary Length  :", len(pdf_summary), "chars")
print("Web Summary Length  :", len(web_summary), "chars")
print("""
Observation:
- PDF content tends to be more structured and formal (ethics frameworks, policy language).
- Web content tends to be more trend-focused, concise, and informal.
- PDF summaries are typically denser with conceptual depth.
- Web summaries are often broader but shallower due to mixed page content (navbars, footers, etc.).
""")