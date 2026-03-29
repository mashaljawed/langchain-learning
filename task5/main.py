import os
from dotenv import load_dotenv

from task2.main import chain_3
from task3.main import retriever

load_dotenv()

# TOOL 1: RETRIEVER
def retrieve_text(query: str) -> str:
    """Retrieve relevant AI breakthrough text from document."""
    docs = retriever.invoke(query)
    return "\n".join([d.page_content for d in docs])


# TOOL 2: SUMMARIZER
def summarize_text(text: str) -> str:
    """Summarizes text using Task 2 chain."""
    return chain_3.invoke({"text": text})


# TOOL 3: WORD COUNTER
def count_words(text: str) -> int:
    """Counts words in text."""
    return len(text.split())


# PIPELINE
def run_pipeline(query: str):
    print("\n=== STEP 1: RETRIEVING ===")
    retrieved = retrieve_text(query)
    print(retrieved[:500], "...")  # preview

    print("\n=== STEP 2: SUMMARIZING ===")
    summary = summarize_text(retrieved)
    print(summary)

    print("\n=== STEP 3: WORD COUNT ===")
    word_count = count_words(summary)
    print("Word Count:", word_count)

    return summary, word_count


# TEST 1
print("\n=== TEST 1 ===")

run_pipeline("Find and summarize text about AI breakthroughs from the document.")


# TEST 2
print("\n=== TEST 2 ===")

run_pipeline("Summarize something interesting")