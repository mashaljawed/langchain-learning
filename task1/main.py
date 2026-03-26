import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("AZURE_OPENAI_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

# Print to confirm they’re loaded
print("API Key:", api_key)
print("Endpoint:", endpoint)
print("Deployment Name:", deployment_name)
print("API Version:", api_version)