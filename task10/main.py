import os
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from task2.main import chain_3
from langchain_community.document_loaders import TextLoader

load_dotenv()

# LLM
llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0
)

# LOAD ai_intro.txt
file_path = os.path.join(os.path.dirname(__file__), "..", "task3", "aiIntro.txt")
loader = TextLoader(file_path, encoding="utf-8")
docs = loader.load()
full_text = docs[0].page_content

# STEP 1: SUMMARIZE FULL DOCUMENT
print("\n=== STEP 1: DOCUMENT SUMMARY ===")
summary = chain_3.invoke({"text": full_text})
print(summary)

# QA CHAIN
qa_prompt = PromptTemplate(
    input_variables=["text", "question"],
    template="""
You are a helpful assistant. Answer the question based only on the provided text.
Be concise and accurate.

Text:
{text}

Question:
{question}

Answer:
"""
)

parser = StrOutputParser()
qa_chain = qa_prompt | llm | parser

# QUESTION
question = "What's the key event mentioned?"

# STEP 2: QA ON SUMMARY
print("\n=== STEP 2: QA ON SUMMARY ===")
summary_answer = qa_chain.invoke({"text": summary, "question": question})
print("Q:", question)
print("A:", summary_answer)

# STEP 3: QA ON FULL DOCUMENT
print("\n=== STEP 3: QA ON FULL DOCUMENT ===")
full_answer = qa_chain.invoke({"text": full_text, "question": question})
print("Q:", question)
print("A:", full_answer)

# COMPARISON
print("\n=== COMPARISON ===")
print("Summary answer length  :", len(summary_answer), "chars")
print("Full doc answer length :", len(full_answer), "chars")

print("\nObservation:")
if len(summary_answer) < len(full_answer):
    print("- Summary-based answer is more concise as expected.")
    print("- Full document answer may include more context and detail.")
elif len(summary_answer) > len(full_answer):
    print("- Surprisingly, summary-based answer is longer than full document answer.")
    print("- Full document may have diluted the key event among other details.")
else:
    print("- Both answers are of similar length.")

print("\nSummary Answer :", summary_answer)
print("Full Doc Answer:", full_answer)