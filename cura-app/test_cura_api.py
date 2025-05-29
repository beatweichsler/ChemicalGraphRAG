# /cura-app/test_azure_openai.py

import os
from openai import AzureOpenAI
from dotenv import load_dotenv

print("--- Starting Azure OpenAI Connection Test ---")

# 1. Load the environment variables from your secrets file
env_path = os.path.join("secrets", "prod.env")
if load_dotenv(env_path):
    print(f"Successfully loaded environment variables from: {env_path}")
else:
    print(f"Error: Could not find or load the file at: {env_path}")
    print("Please make sure the file exists and is in the 'secrets' directory.")
    exit()

# 2. Read the specific variables needed for the test
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("OPENAI_API_VERSION")
deployment_name = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")

# 3. Check if all required variables are present
if not all([endpoint, api_key, api_version, deployment_name]):
    print("\nError: One or more required Azure environment variables are missing.")
    print("Please ensure the following are set in your secrets/prod.env file:")
    print("- AZURE_OPENAI_ENDPOINT")
    print("- AZURE_OPENAI_API_KEY")
    print("- OPENAI_API_VERSION")
    print("- AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
    exit()

print(f"\nAttempting to connect with the following configuration:")
print(f"  Endpoint: {endpoint}")
print(f"  Deployment Name: {deployment_name}")
print(f"  API Version: {api_version}")

try:
    # 4. Initialize the Azure OpenAI client
    client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=api_key,
    api_version="2025-01-01-preview",
)


    # 5. Send a very simple test message
    print("\nSending request to Azure OpenAI...")
    response = client.chat.completions.create(
        model=deployment_name,  # Use the DEPLOYMENT NAME here
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, can you hear me?"},
        ],
        temperature=0.1,
        max_tokens=50
    )

    print("\n--- SUCCESS! ---")
    print("Successfully received a response from Azure.")
    print(f"Response: {response.choices[0].message.content}")

except Exception as e:
    print("\n--- FAILURE! ---")
    print("An error occurred while trying to connect or get a response from Azure.")
    print(f"\nError Type: {type(e).__name__}")
    print(f"Error Details: {e}")