import os
from dotenv import load_dotenv

from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

from task2.main import chain_3

load_dotenv()

# 1. Load file
file_path = os.path.join(os.path.dirname(__file__), "aiIntro.txt")
loader = TextLoader(file_path, encoding="utf-8")
docs = loader.load()

# 2. Split text into chunks
splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)

chunks = splitter.split_documents(docs)

# 3. Embeddings (Azure)
embeddings = AzureOpenAIEmbeddings(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

# 4. Create vector store
vectorstore = FAISS.from_documents(chunks, embeddings)

# 5. Create retriever
retriever = vectorstore.as_retriever()

if __name__ == "__main__":
    # 6. Query
    query = "AI milestones"
    retrieved_docs = retriever.invoke(query)

    retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])

    print("\nRetrieved Context:\n", retrieved_text)

    # 7. Summarize using Task 2 chain
    result = chain_3.invoke({"text": retrieved_text})

    print("\nFinal Summary:\n", result)