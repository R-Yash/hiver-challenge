import json
import chromadb
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from google.genai import types as genai_types
from dotenv import load_dotenv
load_dotenv()

Settings.embed_model = GoogleGenAIEmbedding(
    model_name="gemini-embedding-2",
    embedding_config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT",output_dimensionality=768),
    embed_batch_size=100
)

with open("synthetic_dataset.json", "r") as f:
    dataset = json.load(f)

documents = []

for item in dataset:
    doc = Document(
        text=item["incoming_email"],
        metadata={
            "category": item["category"],
            "reference_reply": item["reference_reply"]
        },
        excluded_embed_metadata_keys=["reference_reply"],
    )
    documents.append(doc)

db_path = "./chroma_db"
chroma_client = chromadb.PersistentClient(path=db_path)
chroma_collection = chroma_client.get_or_create_collection("office_emails")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents, 
    storage_context=storage_context,
    show_progress=True
)
