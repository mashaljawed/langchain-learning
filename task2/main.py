import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
parser = StrOutputParser()

chat_model = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0
)

prompt_3_sentences = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text into exactly 3 sentences:\n\n{text}"
)

chain_3 = prompt_3_sentences | chat_model | parser

prompt_1_sentence = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text into exactly 1 sentence:\n\n{text}"
)

chain_1 = prompt_1_sentence | chat_model | parser

if __name__ == "__main__":
    text = """
    Artificial intelligence (AI) is a branch of computer science that aims to create machines capable of performing tasks that typically require human intelligence. 
    These tasks include problem-solving, understanding natural language, recognizing patterns, and making decisions. 
    Machine learning, a subset of AI, enables systems to learn from data and improve over time without explicit programming. 
    Deep learning, a more advanced form of machine learning, uses neural networks with multiple layers to model complex patterns in large datasets. 
    AI applications are widespread across industries, from healthcare, where it helps in disease diagnosis, to finance, where it aids in fraud detection. 
    Autonomous vehicles leverage AI for navigation and decision-making, improving safety and efficiency. 
    Despite its benefits, AI also poses ethical challenges, such as bias in algorithms and job displacement. 
    Researchers continue to explore ways to make AI more transparent, accountable, and aligned with human values. 
    As technology evolves, AI is expected to play an increasingly significant role in shaping our society, economy, and daily lives.
    """

    result_3 = chain_3.invoke({"text": text})
    print("3-sentence summary:\n", result_3)

    result_1 = chain_1.invoke({"text": text})
    print("\n1-sentence summary:\n", result_1)