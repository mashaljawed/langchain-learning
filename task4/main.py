import os
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool

# ✅ NEW API
from langchain.agents import create_agent

# Import your Task 2 chain
from task2.main import chain_3

load_dotenv()

# 1. LLM
llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0
)

# 2. Tool
@tool
def TextSummarizer(text: str) -> str:
    """Summarizes given text into exactly 3 sentences."""
    return chain_3.invoke({"text": text})

tools = [TextSummarizer]

# 3. agent 
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant that uses tools when needed."
)

#  TEST 1 
text1 = """
Artificial intelligence is transforming healthcare by improving diagnostics,
enabling personalized treatment, and automating administrative processes.
AI systems can analyze medical images with high accuracy and assist doctors
in early disease detection. However, challenges like data privacy and bias
must be addressed.
"""

print("\n=== Test 1: Clear Request ===")
result1 = agent.invoke({
    "messages": [
        {"role": "user", "content": f"Summarize the impact of AI on healthcare:\n{text1}"}
    ]
})
print("\nFinal Answer:\n", result1["messages"][-1].content)


# TEST 2
text2 = """
Space exploration has led to numerous technological advancements, including satellite
communication and GPS systems. It has expanded our understanding of the universe and
inspired generations to pursue science and innovation.
"""

print("\n=== Test 2: Vague Request ===")
result2 = agent.invoke({
    "messages": [
        {"role": "user", "content": f"Summarize something interesting:\n{text2}"}
    ]
})
print("\nFinal Answer:\n", result2["messages"][-1].content)