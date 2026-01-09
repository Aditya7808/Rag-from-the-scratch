"""
HyDE RAG (Hypothetical Document Embeddings) Implementation

This module implements the HyDE retrieval technique which enhances traditional RAG
by generating a hypothetical document first, then using that for similarity search.

Key Concept:
    Instead of embedding the user query directly, HyDE uses an LLM to generate
    a hypothetical answer, embeds that answer, and uses it to find similar documents.
    This bridges the semantic gap between short queries and longer documents.

Author: RAG Documentation Project
Date: January 9, 2026
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np


# ------------------------------------------------------------------------------
# Data Classes
# ------------------------------------------------------------------------------

@dataclass
class Document:
    """
    Represents a document in the knowledge base.
    
    Attributes:
        content: The text content of the document
        metadata: Optional dictionary containing document metadata
        embedding: Optional pre-computed embedding vector
    """
    content: str
    metadata: Optional[Dict[str, Any]] = None
    embedding: Optional[np.ndarray] = None


@dataclass
class RetrievalResult:
    """
    Represents a retrieval result with similarity score.
    
    Attributes:
        document: The retrieved document
        score: Similarity score between query and document
    """
    document: Document
    score: float


# ------------------------------------------------------------------------------
# HyDE RAG Implementation
# ------------------------------------------------------------------------------

class HyDERAG:
    """
    HyDE (Hypothetical Document Embeddings) RAG Implementation.
    
    This class implements the HyDE retrieval technique which generates a
    hypothetical document using an LLM before performing similarity search.
    
    Mathematical Formulation:
        Traditional RAG: R(q) = argmax_k sim(E(q), E(d_i))
        HyDE RAG: R(q) = argmax_k sim(E(LLM(p, q)), E(d_i))
        
        Where:
            - q = user query
            - E = embedding function
            - LLM = language model
            - p = hypothesis generation prompt
            - d_i = documents in corpus
            - sim = similarity function (cosine similarity)
    """
    
    def __init__(
        self,
        llm_client: Any,
        embedding_model: Any,
        vector_store: Any,
        hypothesis_prompt_template: Optional[str] = None
    ):
        """
        Initialize the HyDE RAG system.
        
        Args:
            llm_client: Language model client for generating hypothetical documents
            embedding_model: Model to create vector embeddings
            vector_store: Vector database containing document embeddings
            hypothesis_prompt_template: Custom template for hypothesis generation
        """
        self.llm_client = llm_client
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        
        # Default prompt template for hypothesis generation
        self.hypothesis_prompt_template = hypothesis_prompt_template or """
Please write a detailed passage that would answer the following question.
Write as if this passage exists in a knowledge base or documentation.
Do not include phrases like "I think" or "In my opinion".
Write in a factual, informative tone.

Question: {query}

Passage:
"""
    
    def generate_hypothesis(self, query: str) -> str:
        """
        Generate a hypothetical document that would answer the query.
        
        This is the core innovation of HyDE - instead of searching with the
        original query, we generate what the answer might look like and
        search using that.
        
        Args:
            query: The user's original question
            
        Returns:
            A hypothetical document/passage that would answer the query
        """
        # Format the prompt with the user's query
        prompt = self.hypothesis_prompt_template.format(query=query)
        
        # Generate the hypothetical document using the LLM
        hypothetical_document = self.llm_client.generate(prompt)
        
        return hypothetical_document
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Create an embedding vector for the given text.
        
        Args:
            text: Text to embed (can be query or document)
            
        Returns:
            Embedding vector as numpy array
        """
        embedding = self.embedding_model.encode(text)
        return np.array(embedding)
    
    def cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Formula: sim(a, b) = (a . b) / (||a|| * ||b||)
        
        Args:
            vec_a: First vector
            vec_b: Second vector
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        
        # Avoid division by zero
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def retrieve_documents(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """
        Retrieve the top-k most similar documents from the vector store.
        
        Args:
            query_embedding: The embedding vector to search with
            top_k: Number of documents to retrieve
            
        Returns:
            List of RetrievalResult objects sorted by similarity
        """
        results = self.vector_store.similarity_search(
            embedding=query_embedding,
            top_k=top_k
        )
        return results
    
    def generate_final_answer(
        self,
        query: str,
        retrieved_documents: List[Document]
    ) -> str:
        """
        Generate the final answer using retrieved documents as context.
        
        Args:
            query: The original user query
            retrieved_documents: List of relevant documents from retrieval
            
        Returns:
            The final generated answer
        """
        # Combine retrieved documents into context
        context = "\n\n---\n\n".join([doc.content for doc in retrieved_documents])
        
        # Create the final answer generation prompt
        answer_prompt = f"""
Based on the following context, provide a comprehensive answer to the question.
If the context doesn't contain enough information, say so.

Context:
{context}

Question: {query}

Answer:
"""
        
        # Generate the final answer
        answer = self.llm_client.generate(answer_prompt)
        
        return answer
    
    def query(
        self,
        user_query: str,
        top_k: int = 5,
        return_sources: bool = False
    ) -> Dict[str, Any]:
        """
        Execute the complete HyDE RAG pipeline.
        
        Pipeline Steps:
            1. Generate hypothetical document from query using LLM
            2. Embed the hypothetical document
            3. Retrieve similar documents from vector store
            4. Generate final answer using retrieved context
        
        Args:
            user_query: The user's question
            top_k: Number of documents to retrieve
            return_sources: Whether to include source documents in response
            
        Returns:
            Dictionary containing the answer and optionally source documents
        """
        # Step 1: Generate hypothetical document
        # This transforms the query into a document-like format
        hypothetical_doc = self.generate_hypothesis(user_query)
        
        # Step 2: Embed the hypothetical document
        # We search using this embedding instead of the query embedding
        hypothesis_embedding = self.embed_text(hypothetical_doc)
        
        # Step 3: Retrieve similar documents from vector store
        # Document-to-document matching tends to be more accurate
        retrieval_results = self.retrieve_documents(
            query_embedding=hypothesis_embedding,
            top_k=top_k
        )
        
        # Extract documents from results
        retrieved_docs = [result.document for result in retrieval_results]
        
        # Step 4: Generate final answer using retrieved context
        final_answer = self.generate_final_answer(user_query, retrieved_docs)
        
        # Prepare response
        response = {
            "answer": final_answer,
            "hypothetical_document": hypothetical_doc
        }
        
        if return_sources:
            response["sources"] = [
                {
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "score": result.score
                }
                for doc, result in zip(retrieved_docs, retrieval_results)
            ]
        
        return response


# ------------------------------------------------------------------------------
# Multi-Hypothesis HyDE (Advanced Implementation)
# ------------------------------------------------------------------------------

class MultiHyDERAG(HyDERAG):
    """
    Multi-Hypothesis HyDE RAG Implementation.
    
    This advanced version generates multiple hypothetical documents and
    aggregates their embeddings for more robust retrieval.
    
    Mathematical Formulation:
        H = {h_1, h_2, ..., h_m} = {LLM(p, q, t_j)} for j=1 to m
        e_agg = (1/m) * sum(E(h_j)) for j=1 to m
        R(q) = argmax_k sim(e_agg, E(d_i))
    """
    
    def __init__(
        self,
        llm_client: Any,
        embedding_model: Any,
        vector_store: Any,
        num_hypotheses: int = 3,
        hypothesis_prompt_template: Optional[str] = None
    ):
        """
        Initialize Multi-Hypothesis HyDE RAG.
        
        Args:
            llm_client: Language model client
            embedding_model: Embedding model
            vector_store: Vector database
            num_hypotheses: Number of hypothetical documents to generate
            hypothesis_prompt_template: Custom prompt template
        """
        super().__init__(
            llm_client,
            embedding_model,
            vector_store,
            hypothesis_prompt_template
        )
        self.num_hypotheses = num_hypotheses
    
    def generate_multiple_hypotheses(self, query: str) -> List[str]:
        """
        Generate multiple hypothetical documents for the same query.
        
        Using different sampling (temperature) settings can produce
        diverse hypotheses that cover different aspects of the query.
        
        Args:
            query: The user's question
            
        Returns:
            List of hypothetical documents
        """
        hypotheses = []
        
        for i in range(self.num_hypotheses):
            # Vary temperature for diversity (if supported by LLM client)
            temperature = 0.3 + (i * 0.2)  # 0.3, 0.5, 0.7, ...
            
            prompt = self.hypothesis_prompt_template.format(query=query)
            
            # Generate with different temperature settings
            hypothesis = self.llm_client.generate(
                prompt,
                temperature=temperature
            )
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    def aggregate_embeddings(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Aggregate multiple embeddings into a single embedding.
        
        Uses mean pooling to combine embeddings:
            e_agg = (1/m) * sum(e_j) for j=1 to m
        
        Args:
            embeddings: List of embedding vectors
            
        Returns:
            Aggregated embedding vector
        """
        # Stack embeddings and compute mean
        stacked = np.stack(embeddings)
        aggregated = np.mean(stacked, axis=0)
        
        return aggregated
    
    def query(
        self,
        user_query: str,
        top_k: int = 5,
        return_sources: bool = False
    ) -> Dict[str, Any]:
        """
        Execute Multi-Hypothesis HyDE RAG pipeline.
        
        Args:
            user_query: The user's question
            top_k: Number of documents to retrieve
            return_sources: Whether to include sources in response
            
        Returns:
            Dictionary with answer and optional sources
        """
        # Step 1: Generate multiple hypothetical documents
        hypotheses = self.generate_multiple_hypotheses(user_query)
        
        # Step 2: Embed all hypothetical documents
        hypothesis_embeddings = [
            self.embed_text(hyp) for hyp in hypotheses
        ]
        
        # Step 3: Aggregate embeddings
        aggregated_embedding = self.aggregate_embeddings(hypothesis_embeddings)
        
        # Step 4: Retrieve using aggregated embedding
        retrieval_results = self.retrieve_documents(
            query_embedding=aggregated_embedding,
            top_k=top_k
        )
        
        retrieved_docs = [result.document for result in retrieval_results]
        
        # Step 5: Generate final answer
        final_answer = self.generate_final_answer(user_query, retrieved_docs)
        
        response = {
            "answer": final_answer,
            "hypothetical_documents": hypotheses,
            "num_hypotheses": len(hypotheses)
        }
        
        if return_sources:
            response["sources"] = [
                {
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "score": result.score
                }
                for doc, result in zip(retrieved_docs, retrieval_results)
            ]
        
        return response


# ------------------------------------------------------------------------------
# Example Usage
# ------------------------------------------------------------------------------

def example_usage():
    """
    Example demonstrating how to use the HyDE RAG implementation.
    
    Note: This is pseudocode - actual implementation requires
    real LLM client, embedding model, and vector store instances.
    """
    
    # Initialize components (pseudocode - replace with actual implementations)
    # llm_client = OpenAIClient(api_key="your-api-key")
    # embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    # vector_store = ChromaDB(collection_name="documents")
    
    # Initialize HyDE RAG
    # hyde_rag = HyDERAG(
    #     llm_client=llm_client,
    #     embedding_model=embedding_model,
    #     vector_store=vector_store
    # )
    
    # Execute query
    # result = hyde_rag.query(
    #     user_query="What are the main causes of climate change?",
    #     top_k=5,
    #     return_sources=True
    # )
    
    # Access results
    # print("Answer:", result["answer"])
    # print("Hypothetical Document:", result["hypothetical_document"])
    # if "sources" in result:
    #     for source in result["sources"]:
    #         print(f"Source (score: {source['score']}):", source["content"][:100])
    
    pass


if __name__ == "__main__":
    example_usage()
