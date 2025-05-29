# /cura-app/create_vector_store.py

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# --- Configuration for the New Structure ---
DOCUMENT_SOURCE_PATH = "documents" 
VECTOR_STORE_PATH = "vector_store"
FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_PATH, "faiss_index")
ENV_FILE_PATH = os.path.join("secrets", "prod.env")


def create_index():
    """
    Processes all documents, creates a FAISS vector store using either Azure or Google
    embeddings, and saves it to disk.
    """
    # Load environment variables from the specified .env file
    if not os.path.exists(ENV_FILE_PATH):
        print(f"Error: Environment file not found at '{ENV_FILE_PATH}'.")
        return
        
    load_dotenv(ENV_FILE_PATH)

    # --- Model Selection Logic ---
    print("Checking for embedding model credentials...")
    azure_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    embeddings = None

    if azure_embedding_deployment:
        # Check for all necessary Azure keys if the deployment name is present
        if all([os.getenv("AZURE_OPENAI_ENDPOINT"), os.getenv("AZURE_OPENAI_API_KEY")]):
            print("Found Azure OpenAI credentials. Using Azure for indexing.")
            embeddings = AzureOpenAIEmbeddings(
                azure_deployment=azure_embedding_deployment
            )
        else:
            print("Error: Azure deployment name found, but endpoint or API key is missing.")
            return
            
    elif google_api_key:
        print("Found Google API key. Using Google for indexing.")
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=google_api_key
        )
    else:
        print("Error: No embedding provider credentials found.")
        print("Please provide either GOOGLE_API_KEY or all AZURE_OPENAI_* variables in your secrets/prod.env file.")
        return

    # --- Document Loading and Processing ---
    print(f"Loading all regulation documents from '{DOCUMENT_SOURCE_PATH}' folder...")
    all_chunks = []
    
    if not os.path.isdir(DOCUMENT_SOURCE_PATH):
        print(f"Error: The source directory '{DOCUMENT_SOURCE_PATH}' was not found.")
        return

    supported_files = [f for f in os.listdir(DOCUMENT_SOURCE_PATH) if f.endswith(('.txt', '.pdf'))]
    if not supported_files:
        print(f"No .txt or .pdf files found in '{DOCUMENT_SOURCE_PATH}'. Nothing to index.")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len
    )

    for doc_name in supported_files:
        file_path = os.path.join(DOCUMENT_SOURCE_PATH, doc_name)
        regulation_name = os.path.splitext(doc_name)[0] 
        
        loader = TextLoader(file_path, encoding="utf-8") if doc_name.endswith(".txt") else PyPDFLoader(file_path)
        
        print(f"  - Processing '{doc_name}'...")
        documents = loader.load()
        full_text = "\n".join([doc.page_content for doc in documents])
        
        doc_for_splitting = Document(page_content=full_text, metadata={"source": regulation_name})
        chunks = text_splitter.split_documents([doc_for_splitting])
        all_chunks.extend(chunks)

    if not all_chunks:
        print("Could not create any chunks from the documents. Aborting.")
        return

    print(f"\nTotal chunks created: {len(all_chunks)}")
    print("Generating embeddings and creating FAISS index... (This may take a while)")

    try:
        texts_for_faiss = [chunk.page_content for chunk in all_chunks]
        metadatas_for_faiss = [chunk.metadata for chunk in all_chunks]

        vector_store = FAISS.from_texts(texts_for_faiss, embeddings, metadatas=metadatas_for_faiss)

        os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
        vector_store.save_local(FAISS_INDEX_PATH)
        
        print(f"\nSuccessfully created and saved the FAISS index to '{FAISS_INDEX_PATH}'")

    except Exception as e:
        print(f"\nAn error occurred during indexing: {e}")


if __name__ == '__main__':
    create_index()