"""
RAG (Retrieval-Augmented Generation) System for Guitar Knowledge
Loads structured guitar catalog data from Excel and builds a FAISS vectorstore.
Uses OFFLINE Hugging Face embeddings for reliability.
"""
import os
import ssl
from typing import List
import pandas as pd

# Bypass SSL verification for model downloads (handling enterprise proxy issues)
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Cleaned up environment variables - let pip-system-certs handle it
os.environ['CURL_CA_BUNDLE'] = '' 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from config import (
    RAG_PDF_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_RESULTS,
)


class GuitarKnowledgeRAG:
    """RAG system for guitar shopping knowledge base with keyword fallback"""

    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.documents = None # Store raw documents for fallback
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.load_knowledge_base()

    def _get_embeddings(self):
        """Load offline Hugging Face embeddings"""
        if self.embeddings is None:
            print(f"Loading offline embedding model: {self.model_name}...")
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
            except Exception as e:
                print(f"Error loading offline embeddings: {e}")
                # We don't raise here, we'll use fallback instead
        return self.embeddings

    def load_knowledge_base(self):
        """Load source documents immediately"""
        if not os.path.exists(RAG_PDF_PATH):
            print(f"Warning: Knowledge base file {RAG_PDF_PATH} not found.")
            return

        print(f"Loading documents from {RAG_PDF_PATH}...")
        if RAG_PDF_PATH.endswith(".xlsx") or RAG_PDF_PATH.endswith(".xls"):
            self.documents = self._load_excel_documents()
        else:
            with open(RAG_PDF_PATH, "r", encoding="utf-8") as f:
                text = f.read()
            self.documents = [Document(page_content=text, metadata={"source": RAG_PDF_PATH})]

    def _load_excel_documents(self) -> List[Document]:
        """Load the guitar catalog Excel file and convert each row into a LangChain Document."""
        df = pd.read_excel(RAG_PDF_PATH)
        documents: List[Document] = []

        for _, row in df.iterrows():
            if "full_description" in row and pd.notna(row["full_description"]):
                text = str(row["full_description"])
            else:
                text = self._row_to_text(row)

            # Enrich with structured details
            extras = []
            for col in ["price_usd", "msrp_usd", "price_inr", "brand", "model", "category",
                        "sound_profile", "best_for", "genre_strength", "skill_level",
                        "feel_profile", "recommended_use"]:
                if col in row and pd.notna(row[col]):
                    val = int(row[col]) if "price" in col or "msrp" in col else row[col]
                    prefix = col.replace("_", " ").title()
                    extras.append(f"{prefix}: {val}")

            enriched_text = text + "\n" + " | ".join(extras)
            metadata = {col: str(row[col]) for col in df.columns if pd.notna(row[col])}
            documents.append(Document(page_content=enriched_text, metadata=metadata))

        print(f"Loaded {len(documents)} guitar entries.")
        return documents

    @staticmethod
    def _row_to_text(row) -> str:
        """Fallback: convert a DataFrame row to a descriptive text paragraph."""
        parts = []
        for col in row.index:
            if pd.notna(row[col]):
                parts.append(f"{col}: {row[col]}")
        return " | ".join(parts)

    def _create_vectorstore(self):
        """Create vectorstore (local FAISS index)"""
        if self.vectorstore is not None:
            return

        vectorstore_path = os.path.join(os.path.dirname(RAG_PDF_PATH), "faiss_index_offline")

        # Try to load from local cache first
        if os.path.exists(vectorstore_path):
            try:
                embeddings = self._get_embeddings()
                if embeddings:
                    self.vectorstore = FAISS.load_local(
                        vectorstore_path, embeddings, allow_dangerous_deserialization=True
                    )
                    print(f"Loaded offline vectorstore from {vectorstore_path}")
                    return
            except Exception as e:
                print(f"Could not load local vectorstore: {e}. Recreating...")

        # ---- Build from loaded documents ----
        if not self.documents:
            return

        # Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " | ", " ", ""],
        )
        chunks = splitter.split_documents(self.documents)

        if chunks:
            embeddings = self._get_embeddings()
            if not embeddings:
                print("Failed to load embeddings. Vectorstore creation skipped.")
                return

            print(f"Embedding {len(chunks)} chunks using offline model...")
            self.vectorstore = FAISS.from_documents(chunks, embeddings)

            # Save to local cache
            os.makedirs(os.path.dirname(vectorstore_path), exist_ok=True)
            self.vectorstore.save_local(vectorstore_path)
            print(f"Saved offline vectorstore to {vectorstore_path}")

    def retrieve(self, query: str, k: int = TOP_K_RESULTS) -> List[str]:
        """Retrieve relevant guitar information based on query (with keyword fallback)"""
        # Attempt vector search
        try:
            if self.vectorstore is None:
                self._create_vectorstore()
            
            if self.vectorstore is not None:
                results = self.vectorstore.similarity_search(query, k=k)
                return [doc.page_content for doc in results]
        except Exception as e:
            print(f"Vector retrieval failed: {e}")

        # Fallback: Keyword search
        return self._keyword_search(query, k)

    def _keyword_search(self, query: str, k: int = TOP_K_RESULTS) -> List[str]:
        """Simple keyword search fallback if embeddings/vector DB fail"""
        if not self.documents:
            return ["No catalog data available."]
        
        print(f"Performing keyword fallback for: '{query}'")
        query_words = query.lower().split()
        scores = []
        
        for doc in self.documents:
            content = doc.page_content.lower()
            # Simple count of query words in content
            score = sum(1 for word in query_words if word in content)
            if score > 0:
                scores.append((score, doc.page_content))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[0], reverse=True)
        return [content for score, content in scores[:k]]

    def retrieve_with_context(self, query: str, k: int = 8) -> str:
        """Retrieve and format context for agent use (increased k for better recall)"""
        results = self.retrieve(query, k)
        if not results:
            return "No relevant information found in the knowledge base."

        formatted_results = []
        for i, res in enumerate(results):
            formatted_results.append(f"--- CATALOG ENTRY {i+1} ---\n{res}")
        
        context = "\n\n".join(formatted_results)
        return f"Guitar Catalog Excerpts:\n{context}"


# ------------------------------------------------------------------
# Global singleton
# ------------------------------------------------------------------
_rag_instance = None


def get_rag_system() -> GuitarKnowledgeRAG:
    """Get or create the RAG system instance"""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = GuitarKnowledgeRAG()
    return _rag_instance
