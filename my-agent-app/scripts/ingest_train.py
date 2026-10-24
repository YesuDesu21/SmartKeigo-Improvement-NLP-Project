"""One-off: embed train.jsonl into Pinecone (namespace 'train'). Run from my-agent-app/:
    python -m scripts.ingest_train ../corpus-dataset/splits/train.jsonl
"""
import sys
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
from src.agent.retrieval import Retriever

if __name__ == "__main__":
    n = Retriever().ingest_train(sys.argv[1])
    print(f"Ingested {n} train examples")
