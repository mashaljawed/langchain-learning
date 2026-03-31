import os
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

# LLM
llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0
)

# OUTPUT SCHEMA (Pydantic)
class SummaryOutput(BaseModel):
    summary: str = Field(description="A 3-sentence summary of the input text.")
    length: int = Field(description="The character count of the summary as an integer.")

# PARSER
parser = PydanticOutputParser(pydantic_object=SummaryOutput)
format_instructions = parser.get_format_instructions()

# PROMPT
prompt = PromptTemplate(
    input_variables=["text"],
    partial_variables={"format_instructions": format_instructions},
    template="""
Summarize the following text into exactly 3 sentences.
Return the summary and its character count.

{format_instructions}

Text:
{text}
"""
)

# CHAIN
chain = prompt | llm | parser

# TEST TEXT 
text = """
Artificial intelligence is being applied across numerous industries to solve complex problems.
In healthcare, AI assists in early disease detection and personalized medicine.
In finance, it powers fraud detection systems and algorithmic trading.
In education, AI enables adaptive learning platforms that cater to individual student needs.
Natural language processing allows machines to understand and generate human language,
enabling applications like chatbots and translation services.
Computer vision enables machines to interpret visual data, supporting autonomous vehicles
and facial recognition. Despite these advances, AI raises concerns about job displacement,
data privacy, and ethical decision-making that society must address thoughtfully.
"""

# RUN
print("\n=== Task 8: Structured Output Parser ===\n")

result = chain.invoke({"text": text})

print("Type   :", type(result))
print("Output :", result)
print("\nSummary:\n", result.summary)
print("\nLength (from model):", result.length)
print("Length (verified)  :", len(result.summary))

# VALIDATION
print("\n=== VALIDATION ===")
print("Is SummaryOutput   :", isinstance(result, SummaryOutput))
print("Has summary field  :", hasattr(result, "summary"))
print("Has length field   :", hasattr(result, "length"))