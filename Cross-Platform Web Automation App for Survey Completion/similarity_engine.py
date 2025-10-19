"""
Similarity Engine
Core engine for computing semantic similarity between questions using embeddings.
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from sentence_transformers import SentenceTransformer, util
import pickle
import json
from pathlib import Path


class SimilarityEngine:
    """
    Handles embedding generation and similarity computation for question matching.
    Uses SentenceTransformers for high-quality semantic embeddings.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: Optional[str] = None):
        """
        Initialize the Similarity Engine.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
            cache_dir: Directory to cache embeddings and models
        """
        self.logger = logging.getLogger("similarity_engine")
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./cache/embeddings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize the embedding model
        self.model: Optional[SentenceTransformer] = None
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.cache_file = self.cache_dir / "embedding_cache.pkl"
        
        # Load cached embeddings if available
        self._load_embedding_cache()

    async def initialize(self) -> bool:
        """Initialize the embedding model."""
        try:
            self.logger.info(f"Loading SentenceTransformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.logger.info("Similarity Engine initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Similarity Engine: {e}", exc_info=True)
            return False

    def encode_questions(self, questions: List[str], use_cache: bool = True) -> np.ndarray:
        """
        Encode a list of questions into embeddings.
        
        Args:
            questions: List of question strings
            use_cache: Whether to use cached embeddings
            
        Returns:
            Array of embeddings
        """
        if not self.model:
            raise RuntimeError("Similarity Engine not initialized")
        
        embeddings = []
        questions_to_encode = []
        indices_to_encode = []
        
        for i, question in enumerate(questions):
            if use_cache and question in self.embedding_cache:
                embeddings.append(self.embedding_cache[question])
            else:
                embeddings.append(None)  # Placeholder
                questions_to_encode.append(question)
                indices_to_encode.append(i)
        
        # Encode questions not in cache
        if questions_to_encode:
            self.logger.debug(f"Encoding {len(questions_to_encode)} new questions")
            new_embeddings = self.model.encode(questions_to_encode, convert_to_tensor=False)
            
            # Update cache and embeddings list
            for i, (question, embedding) in enumerate(zip(questions_to_encode, new_embeddings)):
                self.embedding_cache[question] = embedding
                embeddings[indices_to_encode[i]] = embedding
            
            # Save updated cache
            if use_cache:
                self._save_embedding_cache()
        
        return np.array(embeddings)

    def compute_similarity(self, query_embedding: np.ndarray, 
                          candidate_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between query and candidate embeddings.
        
        Args:
            query_embedding: Single query embedding
            candidate_embeddings: Array of candidate embeddings
            
        Returns:
            Array of similarity scores
        """
        # Ensure embeddings are 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        if candidate_embeddings.ndim == 1:
            candidate_embeddings = candidate_embeddings.reshape(1, -1)
        
        # Compute cosine similarity
        similarities = util.cos_sim(query_embedding, candidate_embeddings)[0]
        return similarities.cpu().numpy() if hasattr(similarities, 'cpu') else similarities

    def find_most_similar(self, query_question: str, 
                         candidate_questions: List[str],
                         threshold: float = 0.7) -> List[Tuple[int, str, float]]:
        """
        Find the most similar questions to a query question.
        
        Args:
            query_question: Question to find matches for
            candidate_questions: List of candidate questions
            threshold: Minimum similarity threshold
            
        Returns:
            List of tuples (index, question, similarity_score) sorted by similarity
        """
        if not candidate_questions:
            return []
        
        # Encode all questions
        all_questions = [query_question] + candidate_questions
        embeddings = self.encode_questions(all_questions)
        
        query_embedding = embeddings[0]
        candidate_embeddings = embeddings[1:]
        
        # Compute similarities
        similarities = self.compute_similarity(query_embedding, candidate_embeddings)
        
        # Filter by threshold and sort
        results = []
        for i, (question, similarity) in enumerate(zip(candidate_questions, similarities)):
            if similarity >= threshold:
                results.append((i, question, float(similarity)))
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x[2], reverse=True)
        
        return results

    def batch_similarity_matrix(self, questions: List[str]) -> np.ndarray:
        """
        Compute pairwise similarity matrix for a list of questions.
        
        Args:
            questions: List of questions
            
        Returns:
            Similarity matrix (n x n)
        """
        embeddings = self.encode_questions(questions)
        similarity_matrix = util.cos_sim(embeddings, embeddings)
        
        return similarity_matrix.cpu().numpy() if hasattr(similarity_matrix, 'cpu') else similarity_matrix

    def cluster_similar_questions(self, questions: List[str], 
                                 similarity_threshold: float = 0.8) -> List[List[int]]:
        """
        Group questions into clusters based on similarity.
        
        Args:
            questions: List of questions to cluster
            similarity_threshold: Minimum similarity for clustering
            
        Returns:
            List of clusters, where each cluster is a list of question indices
        """
        if len(questions) < 2:
            return [[i] for i in range(len(questions))]
        
        # Compute similarity matrix
        similarity_matrix = self.batch_similarity_matrix(questions)
        
        # Simple clustering algorithm
        clusters = []
        assigned = set()
        
        for i in range(len(questions)):
            if i in assigned:
                continue
            
            # Start new cluster
            cluster = [i]
            assigned.add(i)
            
            # Find similar questions
            for j in range(i + 1, len(questions)):
                if j not in assigned and similarity_matrix[i][j] >= similarity_threshold:
                    cluster.append(j)
                    assigned.add(j)
            
            clusters.append(cluster)
        
        return clusters

    def get_embedding_stats(self) -> Dict[str, Any]:
        """
        Get statistics about cached embeddings.
        
        Returns:
            Dictionary with embedding statistics
        """
        return {
            "cached_embeddings": len(self.embedding_cache),
            "model_name": self.model_name,
            "cache_file": str(self.cache_file),
            "embedding_dimension": len(next(iter(self.embedding_cache.values()))) if self.embedding_cache else 0
        }

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self.embedding_cache.clear()
        if self.cache_file.exists():
            self.cache_file.unlink()
        self.logger.info("Embedding cache cleared")

    def _load_embedding_cache(self) -> None:
        """Load cached embeddings from disk."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                self.logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")
        except Exception as e:
            self.logger.warning(f"Failed to load embedding cache: {e}")
            self.embedding_cache = {}

    def _save_embedding_cache(self) -> None:
        """Save embeddings to cache file."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            self.logger.debug(f"Saved {len(self.embedding_cache)} embeddings to cache")
        except Exception as e:
            self.logger.warning(f"Failed to save embedding cache: {e}")

    def export_embeddings(self, output_file: str, format: str = "json") -> bool:
        """
        Export cached embeddings to a file.
        
        Args:
            output_file: Path to output file
            format: Export format ("json", "pickle", "numpy")
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if format == "json":
                # Convert numpy arrays to lists for JSON serialization
                json_data = {
                    question: embedding.tolist() 
                    for question, embedding in self.embedding_cache.items()
                }
                with open(output_file, 'w') as f:
                    json.dump(json_data, f, indent=2)
                    
            elif format == "pickle":
                with open(output_file, 'wb') as f:
                    pickle.dump(self.embedding_cache, f)
                    
            elif format == "numpy":
                questions = list(self.embedding_cache.keys())
                embeddings = np.array(list(self.embedding_cache.values()))
                np.savez(output_file, questions=questions, embeddings=embeddings)
                
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Exported {len(self.embedding_cache)} embeddings to {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export embeddings: {e}", exc_info=True)
            return False

    def import_embeddings(self, input_file: str, format: str = "json") -> bool:
        """
        Import embeddings from a file.
        
        Args:
            input_file: Path to input file
            format: Import format ("json", "pickle", "numpy")
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if format == "json":
                with open(input_file, 'r') as f:
                    json_data = json.load(f)
                # Convert lists back to numpy arrays
                self.embedding_cache.update({
                    question: np.array(embedding) 
                    for question, embedding in json_data.items()
                })
                
            elif format == "pickle":
                with open(input_file, 'rb') as f:
                    imported_cache = pickle.load(f)
                self.embedding_cache.update(imported_cache)
                
            elif format == "numpy":
                data = np.load(input_file)
                questions = data['questions']
                embeddings = data['embeddings']
                for question, embedding in zip(questions, embeddings):
                    self.embedding_cache[question] = embedding
                    
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Imported embeddings from {input_file}")
            self._save_embedding_cache()  # Save to local cache
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to import embeddings: {e}", exc_info=True)
            return False

