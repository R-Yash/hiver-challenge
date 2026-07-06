import chromadb
from llama_index.core import VectorStoreIndex, Settings, PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from google.genai import types as genai_types
from dotenv import load_dotenv
load_dotenv()

Settings.llm = GoogleGenAI(model="gemini-2.5-flash", temperature=0.3)
Settings.embed_model = GoogleGenAIEmbedding(
    model_name="gemini-embedding-2",
    embedding_config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT",output_dimensionality=768),
    embed_batch_size=100
)

db_path = "./chroma_db"
chroma_client = chromadb.PersistentClient(path=db_path)
chroma_collection = chroma_client.get_collection("office_emails")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

def generator(email: str) -> str:
    

    email_qa_prompt = PromptTemplate(
        "You are an AI email assistant for internal company communications.\n"
        "Below are historical examples of similar incoming emails and the exact replies our team sent.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the historical context above, draft a professional reply to the new email below.\n"
        "Pay strict attention to formatting, tone, and any company policies (SOPs) demonstrated in the historical replies "
        "(e.g., asking for Jira tickets, proposing 2 time slots, reminding about OOO calendars, or reviewing PRs by EOD).\n\n"
        "New Incoming Email:\n"
        "{query_str}\n\n"
        "Suggested Reply:"
    )

    query_engine = index.as_query_engine(
        similarity_top_k=3,
        text_qa_template=email_qa_prompt
    )
    
    return query_engine.query(email)

if __name__ == "__main__":
    # email = input("Enter the new incoming email: ")
    response = generator("Hey team, my monitor just went completely black and won't turn back on. I tried swapping cables. Can someone from IT replace this?")

    print(response.response)
    
    print("\n--- RETRIEVED CONTEXT USED (For Debugging) ---")
    for node in response.source_nodes:
        print(f"Similarity Score: {node.score:.4f}")
        print(f"Metadata: {node.metadata}")
        print("-" * 20)