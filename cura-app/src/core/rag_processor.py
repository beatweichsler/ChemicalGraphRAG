import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import fitz # PyMuPDF for PDF processing if using custom logic

# --- Configuration ---
# These paths are relative to the project root, where the app will be run from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VECTOR_STORE_PATH = os.path.join(PROJECT_ROOT, "vector_store")
FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_PATH, "faiss_index")

# --- Global objects to load models and vector store only once ---
LLM = None
VECTOR_STORE = None

def load_models_and_store():
    """
    Loads the LLM and the FAISS vector store from disk.
    This function is called once to initialize the global variables.
    It assumes GOOGLE_API_KEY is already loaded in the environment.
    """
    global LLM, VECTOR_STORE
    
    # Check if models are already loaded to prevent reloading
    if LLM is not None and VECTOR_STORE is not None:
        return

        print("Initializing models and loading vector store for RAG...")
    
    # --- Model Selection Logic ---
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
    azure_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
    google_api_key = os.getenv("GOOGLE_API_KEY")

    embeddings = None
    
    if all([azure_endpoint, azure_api_key, azure_chat_deployment, azure_embedding_deployment]):
        print("Found Azure OpenAI credentials. Initializing Azure models for RAG.")
        LLM = AzureChatOpenAI(
            azure_deployment=azure_chat_deployment,
            temperature=0.2
        )
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=azure_embedding_deployment
        )
    elif google_api_key:
        print("Found Google API key. Initializing Google Gemini models for RAG.")
        LLM = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            google_api_key=google_api_key, 
            temperature=0.2
        )
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=google_api_key
        )
    else:
        raise ValueError("LLM provider credentials not found. Please provide either GOOGLE_API_KEY or all AZURE_OPENAI_* variables in your environment.")
    
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index not found at '{FAISS_INDEX_PATH}'. "
            "Please run the `create_vector_store.py` script from the project root first."
        )
    
    VECTOR_STORE = FAISS.load_local(
        FAISS_INDEX_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True # Required for FAISS with pickle
    )
    print("Models and vector store initialized successfully.")


def get_regulation_summary(regulation_name: str, chemical_name: str) -> str:
    """
    Performs RAG using a pre-built vector store to summarize how a chemical is regulated.
    """
    try:
        # This will only run the loading logic on the very first call
        load_models_and_store()
    except (ValueError, FileNotFoundError) as e:
        return str(e)

    prompt_template_str = """
    You are an assistant specialized in chemical regulations.
    Based ONLY on the following text excerpts from the regulation document '{regulation_name}',
    provide a concise summary of how the chemical '{chemical_name}' is specifically mentioned or regulated.
    Focus on:
    - Restrictions, bans, or prohibitions.
    - Concentration limits or thresholds.
    - Specific conditions of use or exemptions.
    - Reporting or notification requirements related to this chemical.
    - If the chemical is listed as a substance of concern.

    If the chemical '{chemical_name}' is not mentioned or no specific regulatory details about it are found in the provided text,
    clearly state that. Do not infer information beyond the provided context.

    Context from document:
    {context}

    Concise Summary for '{chemical_name}' in '{regulation_name}':
    """
    prompt = PromptTemplate(
        template=prompt_template_str,
        input_variables=["context", "chemical_name", "regulation_name"]
    )

    retriever = VECTOR_STORE.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 5, 'filter': {'source': regulation_name}}
    )
    
    query_for_retriever = f"Information about {chemical_name} regulation"
    
    try:
        relevant_docs = retriever.get_relevant_documents(query_for_retriever)
        
        if not relevant_docs:
            return f"No relevant information found for '{chemical_name}' in the document for '{regulation_name}'."

        chain = load_qa_chain(LLM, chain_type="stuff", prompt=prompt) 
        
        result = chain.invoke({
            "input_documents": relevant_docs, 
            "chemical_name": chemical_name,
            "regulation_name": regulation_name
        }, return_only_outputs=True)
        
        return result.get("output_text", "No summary could be generated.")
        
    except Exception as e:
        return f"Error during RAG chain execution: {e}"