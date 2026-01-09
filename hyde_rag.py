"""
HyDE RAG (Hypothetical Document Embeddings) Implementation
Generates a hypothetical answer first, then uses it for similarity search.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class Document:
    content: str
    metadata: Optional[Dict[str, Any]] = None
    embedding: Optional[np.ndarray] = None


@dataclass
class RetrievalResult:
    document: Document
    score: float


class HyDERAG:
    """
    HyDE RAG: R(q) = argmax_k sim(E(LLM(p, q)), E(d_i))
    Instead of embedding query directly, generates hypothetical answer first.
    """
    
    def __init__(
        self,
        llm_client: Any,
        embedding_model: Any,
        vector_store: Any,
        hypothesis_prompt_template: Optional[str] = None
    ):
        self.llm_client = llm_client
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        
        self.hypothesis_prompt_template = hypothesis_prompt_template or """
Please write a detailed passage that would answer the following question.
Write in a factual, informative tone.

Question: {query}

Passage:
"""
    
    def generate_hypothesis(self, query: str) -> str:
        """Generate hypothetical document that would answer the query."""
        prompt = self.hypothesis_prompt_template.format(query=query)
        return self.llm_client.generate(prompt)
    
    def embed_text(self, text: str) -> np.ndarray:
        """Create embedding vector for given text."""
        return np.array(self.embedding_model.encode(text))
    
    def cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Calculate cosine similarity: sim(a,b) = (a.b) / (||a|| * ||b||)"""
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)
    
    def retrieve_documents(self, query_embedding: np.ndarray, top_k: int = 5) -> List[RetrievalResult]:
        """Retrieve top-k most similar documents from vector store."""
        return self.vector_store.similarity_search(embedding=query_embedding, top_k=top_k)
    
    def generate_final_answer(self, query: str, retrieved_documents: List[Document]) -> str:
        """Generate final answer using retrieved documents as context."""
        context = "\n\n---\n\n".join([doc.content for doc in retrieved_documents])
        
        answer_prompt = f"""
Based on the following context, provide a comprehensive answer to the question.

Context:
{context}

Question: {query}

Answer:
"""
        return self.llm_client.generate(answer_prompt)
    
    def query(self, user_query: str, top_k: int = 5, return_sources: bool = False) -> Dict[str, Any]:
        """
        Execute HyDE RAG pipeline:
        1. Generate hypothetical document
        2. Embed hypothetical document
        3. Retrieve similar documents
        4. Generate final answer
        """
        hypothetical_doc = self.generate_hypothesis(user_query)
        hypothesis_embedding = self.embed_text(hypothetical_doc)
        retrieval_results = self.retrieve_documents(hypothesis_embedding, top_k)
        retrieved_docs = [result.document for result in retrieval_results]
        final_answer = self.generate_final_answer(user_query, retrieved_docs)
        
        response = {"answer": final_answer, "hypothetical_document": hypothetical_doc}
        
        if return_sources:
            response["sources"] = [
                {"content": doc.content, "metadata": doc.metadata, "score": result.score}
                for doc, result in zip(retrieved_docs, retrieval_results)
            ]
        return response


class MultiHyDERAG(HyDERAG):
    """
    Multi-Hypothesis HyDE: Generates multiple hypothetical documents and aggregates embeddings.
    e_agg = (1/m) * sum(E(h_j)) for j=1 to m
    """
    
    def __init__(
        self,
        llm_client: Any,
        embedding_model: Any,
        vector_store: Any,
        num_hypotheses: int = 3,
        hypothesis_prompt_template: Optional[str] = None
    ):
        super().__init__(llm_client, embedding_model, vector_store, hypothesis_prompt_template)
        self.num_hypotheses = num_hypotheses
    
    def generate_multiple_hypotheses(self, query: str) -> List[str]:
        """Generate multiple hypothetical documents with varying temperatures."""
        hypotheses = []
        for i in range(self.num_hypotheses):
            temperature = 0.3 + (i * 0.2)
            prompt = self.hypothesis_prompt_template.format(query=query)
            hypothesis = self.llm_client.generate(prompt, temperature=temperature)
            hypotheses.append(hypothesis)
        return hypotheses
    
    def aggregate_embeddings(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """Aggregate embeddings using mean pooling."""
        return np.mean(np.stack(embeddings), axis=0)
    
    def query(self, user_query: str, top_k: int = 5, return_sources: bool = False) -> Dict[str, Any]:
        """Execute Multi-Hypothesis HyDE RAG pipeline."""
        hypotheses = self.generate_multiple_hypotheses(user_query)
        hypothesis_embeddings = [self.embed_text(hyp) for hyp in hypotheses]
        aggregated_embedding = self.aggregate_embeddings(hypothesis_embeddings)
        
        retrieval_results = self.retrieve_documents(aggregated_embedding, top_k)
        retrieved_docs = [result.document for result in retrieval_results]
        final_answer = self.generate_final_answer(user_query, retrieved_docs)
        
        response = {
            "answer": final_answer,
            "hypothetical_documents": hypotheses,
            "num_hypotheses": len(hypotheses)
        }
        
        if return_sources:
            response["sources"] = [
                {"content": doc.content, "metadata": doc.metadata, "score": result.score}
                for doc, result in zip(retrieved_docs, retrieval_results)
            ]
        return response


if __name__ == "__main__":
    # Example usage (pseudocode):
    # llm_client = OpenAIClient(api_key="your-api-key")
    # embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    # vector_store = ChromaDB(collection_name="documents")
    # hyde_rag = HyDERAG(llm_client, embedding_model, vector_store)
    # result = hyde_rag.query("What causes diabetes?", top_k=5)
    pass
