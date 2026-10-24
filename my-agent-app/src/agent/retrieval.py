"""Train-only retrieval over Pinecone with Cohere embeddings.

Only files named train*.jsonl may be ingested, so test/val items can never reach the index.
"""
import json, os
from pathlib import Path
from typing import Dict, List, Optional

INDEX_NAME = "keigo-train"
NAMESPACE = "train"
EMBED_MODEL = "embed-multilingual-v3.0"


def _vecs(resp):
    emb = resp.embeddings
    return getattr(emb, "float_", None) or getattr(emb, "float")


class Retriever:
    def __init__(self):
        import cohere
        from pinecone import Pinecone
        self.co = cohere.ClientV2(os.environ["COHERE_API_KEY"])
        self.pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    def ensure_index(self):
        from pinecone import ServerlessSpec
        if INDEX_NAME not in self.pc.list_indexes().names():
            self.pc.create_index(name=INDEX_NAME, dimension=1024, metric="cosine",
                                 spec=ServerlessSpec(cloud="aws", region="us-east-1"))
        return self.pc.Index(INDEX_NAME)

    def ingest_train(self, path: str, only_core: bool = True, batch: int = 90) -> int:
        if not Path(path).name.startswith("train"):
            raise ValueError("Refusing to ingest a non-train file (leakage guard)")
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        rows = [r for r in rows if r["register"] in ("teineigo", "sonkeigo", "kenjogo") and (r["sft_core"] or not only_core)]
        index = self.ensure_index()
        for i in range(0, len(rows), batch):
            chunk = rows[i:i + batch]
            vs = _vecs(self.co.embed(texts=[r["source_text"] for r in chunk], model=EMBED_MODEL,
                                     input_type="search_document", embedding_types=["float"]))
            index.upsert(namespace=NAMESPACE, vectors=[
                {"id": r["id"], "values": v,
                 "metadata": {"source_text": r["source_text"], "target_text": r["target_text"], "register": r["register"]}}
                for r, v in zip(chunk, vs)])
        return len(rows)

    def retrieve(self, query: str, top_k: int = 3, register: Optional[str] = None) -> List[Dict[str, str]]:
        v = _vecs(self.co.embed(texts=[query], model=EMBED_MODEL, input_type="search_query",
                                embedding_types=["float"]))[0]
        index = self.pc.Index(INDEX_NAME)
        flt = {"register": {"$eq": register}} if register in ("teineigo", "sonkeigo", "kenjogo") else None
        res = index.query(vector=v, top_k=top_k + 1, include_metadata=True, namespace=NAMESPACE, filter=flt)
        out = []
        for m in res["matches"]:
            md = m["metadata"]
            if md["source_text"] == query:   # never return the query itself
                continue
            out.append({"id": m["id"], **md})
        return out[:top_k]
