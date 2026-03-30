import os
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import MultiQueryRetriever

from task2.main import chain_3

load_dotenv()

# LLM
llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0
)

# EMBEDDINGS
embeddings = AzureOpenAIEmbeddings(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

# LOAD & SPLIT ai_intro.txt (same as Task 3)
file_path = os.path.join(os.path.dirname(__file__), "..", "task3", "aiIntro.txt")
loader = TextLoader(file_path, encoding="utf-8")
docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)
chunks = splitter.split_documents(docs)

# VECTOR STORE
vectorstore = FAISS.from_documents(chunks, embeddings)

# SINGLE QUERY RETRIEVER (Task 3 style)
single_retriever = vectorstore.as_retriever()

# MULTI QUERY RETRIEVER
multi_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm
)

# QUERY
query = "AI advancements"

# SINGLE QUERY RESULTS
print("\n=== SINGLE QUERY RETRIEVER ===")
single_docs = single_retriever.invoke(query)
single_text = "\n".join([doc.page_content for doc in single_docs])
print(f"Chunks retrieved: {len(single_docs)}")
print(single_text[:500], "...")

#  MULTI QUERY RESULTS
print("\n=== MULTI QUERY RETRIEVER ===")
multi_docs = multi_retriever.invoke(query)
multi_text = "\n".join([doc.page_content for doc in multi_docs])
print(f"Chunks retrieved: {len(multi_docs)}")
print(multi_text[:500], "...")

# SUMMARIZE BOTH
print("\n=== SINGLE QUERY SUMMARY ===")
single_summary = chain_3.invoke({"text": single_text})
print(single_summary)

print("\n=== MULTI QUERY SUMMARY ===")
multi_summary = chain_3.invoke({"text": multi_text})
print(multi_summary)

# COMPARISON
print("\n=== COMPARISON ===")
print("Single - Chunks retrieved:", len(single_docs))
print("Multi  - Chunks retrieved:", len(multi_docs))
print("Single - Summary length  :", len(single_summary), "chars")
print("Multi  - Summary length  :", len(multi_summary), "chars")

print("\nObservation:")
if len(multi_docs) > len(single_docs):
    print(f"- MultiQueryRetriever retrieved {len(multi_docs) - len(single_docs)} more chunk(s) than the single retriever.")
    print("- The broader retrieval likely produced a more diverse and comprehensive summary.")
elif len(multi_docs) == len(single_docs):
    print("- Both retrievers returned the same number of chunks.")
    print("- This can happen when the document is small and all alternate queries map to the same chunks.")
    print("- With a larger document, MultiQueryRetriever would typically retrieve more diverse chunks.")
else:
    print("- Single retriever returned more chunks than multi — unexpected, may indicate deduplication reduced results.")

if len(multi_summary) > len(single_summary):
    print("- Multi-query summary is longer, suggesting broader coverage.")
elif len(multi_summary) < len(single_summary):
    print("- Single-query summary is longer in this case.")
else:
    print("- Both summaries are of equal length.")