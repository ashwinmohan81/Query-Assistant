import os

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" | "bedrock" | "mock"

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0")
BEDROCK_REGION = os.getenv("BEDROCK_REGION", "us-east-1")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CATALOG_PATH = os.getenv("CATALOG_PATH", "./data/catalog.json")

TOP_K_PRODUCTS = 3
TOP_K_ATTRIBUTES = 20
TOP_K_EXAMPLES = 3
MAX_VALIDATION_RETRIES = 2
RELATIONSHIP_RELEVANCE_THRESHOLD = 0.25
