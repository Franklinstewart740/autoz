"""
Question Mapper
Combines similarity engine and template manager to provide semantic anchoring for questions.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import asyncio

from .similarity_engine import SimilarityEngine
from .template_manager import TemplateManager, QuestionTemplate


class QuestionMapper:
    """
    Maps survey questions to known templates using semantic similarity.
    Enables reuse of fine-tuned response logic for similar questions.
    """

    def __init__(self, 
                 similarity_engine: Optional[SimilarityEngine] = None,
                 template_manager: Optional[TemplateManager] = None,
                 similarity_threshold: float = 0.7):
        """
        Initialize the Question Mapper.
        
        Args:
            similarity_engine: SimilarityEngine instance
            template_manager: TemplateManager instance
            similarity_threshold: Minimum similarity score for matching
        """
        self.logger = logging.getLogger("question_mapper")
        
        # Initialize components if not provided
        self.similarity_engine = similarity_engine or SimilarityEngine()
        self.template_manager = template_manager or TemplateManager()
        self.similarity_threshold = similarity_threshold
        
        # Cache for template embeddings
        self._template_embeddings_cache = None
        self._template_texts_cache = None
        self._template_ids_cache = None

    async def initialize(self) -> bool:
        """Initialize the Question Mapper and its components."""
        try:
            self.logger.info("Initializing Question Mapper")
            
            # Initialize similarity engine
            if not await self.similarity_engine.initialize():
                return False
            
            # Precompute template embeddings
            await self._update_template_embeddings()
            
            self.logger.info("Question Mapper initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Question Mapper: {e}", exc_info=True)
            return False

    async def map_question(self, question_text: str, 
                          question_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Map a question to the most similar template.
        
        Args:
            question_text: The question text to map
            question_context: Additional context about the question
            
        Returns:
            Dictionary with mapping results
        """
        try:
            # Ensure template embeddings are up to date
            await self._ensure_template_embeddings()
            
            if not self._template_texts_cache:
                self.logger.warning("No templates available for mapping")
                return {
                    "success": False,
                    "error": "No templates available",
                    "question": question_text
                }
            
            # Find most similar templates
            similar_templates = self.similarity_engine.find_most_similar(
                question_text,
                self._template_texts_cache,
                threshold=self.similarity_threshold
            )
            
            if not similar_templates:
                self.logger.debug(f"No similar templates found for: {question_text}")
                return {
                    "success": True,
                    "question": question_text,
                    "matched_template": None,
                    "similarity_score": 0.0,
                    "alternatives": []
                }
            
            # Get the best match
            best_match_idx, best_match_text, best_similarity = similar_templates[0]
            best_template_id = self._template_ids_cache[best_match_idx]
            best_template = self.template_manager.get_template(best_template_id)
            
            # Increment usage count
            self.template_manager.increment_usage(best_template_id)
            
            # Prepare alternatives
            alternatives = []
            for idx, text, similarity in similar_templates[1:5]:  # Top 5 alternatives
                template_id = self._template_ids_cache[idx]
                template = self.template_manager.get_template(template_id)
                alternatives.append({
                    "template_id": template_id,
                    "template_text": text,
                    "similarity_score": similarity,
                    "category": template.category if template else "unknown",
                    "question_type": template.question_type if template else "unknown"
                })
            
            self.logger.info(f"Mapped question to template '{best_template_id}' with similarity {best_similarity:.3f}")
            
            return {
                "success": True,
                "question": question_text,
                "matched_template": {
                    "template_id": best_template_id,
                    "template_text": best_match_text,
                    "template_object": best_template,
                    "response_strategy": best_template.response_strategy if best_template else {}
                },
                "similarity_score": best_similarity,
                "alternatives": alternatives
            }
            
        except Exception as e:
            self.logger.error(f"Error mapping question '{question_text}': {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "question": question_text
            }

    async def map_questions_batch(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        Map multiple questions to templates in batch.
        
        Args:
            questions: List of question texts to map
            
        Returns:
            List of mapping results
        """
        results = []
        for question in questions:
            result = await self.map_question(question)
            results.append(result)
        return results

    async def suggest_new_template(self, question_text: str, 
                                  category: str,
                                  question_type: str,
                                  response_strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Suggest creating a new template for a question that doesn't match existing ones.
        
        Args:
            question_text: The question text
            category: Category for the new template
            question_type: Question type
            response_strategy: Response strategy configuration
            
        Returns:
            Dictionary with suggestion results
        """
        try:
            # Check if a very similar template already exists
            await self._ensure_template_embeddings()
            
            if self._template_texts_cache:
                similar_templates = self.similarity_engine.find_most_similar(
                    question_text,
                    self._template_texts_cache,
                    threshold=0.9  # High threshold for duplicate detection
                )
                
                if similar_templates:
                    best_match_idx, best_match_text, best_similarity = similar_templates[0]
                    best_template_id = self._template_ids_cache[best_match_idx]
                    
                    return {
                        "success": False,
                        "reason": "Similar template already exists",
                        "existing_template": {
                            "template_id": best_template_id,
                            "template_text": best_match_text,
                            "similarity_score": best_similarity
                        }
                    }
            
            # Generate template ID
            import hashlib
            template_id = hashlib.md5(question_text.encode()).hexdigest()[:8]
            
            # Create template suggestion
            from datetime import datetime
            suggested_template = QuestionTemplate(
                id=f"template_{template_id}",
                template_text=question_text,
                category=category,
                question_type=question_type,
                response_strategy=response_strategy,
                examples=[question_text],
                tags=[category, question_type],
                confidence_threshold=0.7,
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat()
            )
            
            return {
                "success": True,
                "suggested_template": suggested_template,
                "template_id": suggested_template.id
            }
            
        except Exception as e:
            self.logger.error(f"Error suggesting template for '{question_text}': {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }

    async def add_template_from_question(self, question_text: str,
                                       category: str,
                                       question_type: str,
                                       response_strategy: Dict[str, Any],
                                       examples: Optional[List[str]] = None,
                                       tags: Optional[List[str]] = None) -> bool:
        """
        Create and add a new template based on a question.
        
        Args:
            question_text: The question text to use as template
            category: Template category
            question_type: Question type
            response_strategy: Response strategy configuration
            examples: Additional example questions
            tags: Template tags
            
        Returns:
            True if successful, False otherwise
        """
        try:
            suggestion = await self.suggest_new_template(
                question_text, category, question_type, response_strategy
            )
            
            if not suggestion["success"]:
                self.logger.warning(f"Cannot add template: {suggestion.get('reason', 'Unknown error')}")
                return False
            
            template = suggestion["suggested_template"]
            
            # Update with provided examples and tags
            if examples:
                template.examples.extend(examples)
            if tags:
                template.tags.extend(tags)
            
            # Add template to manager
            success = self.template_manager.add_template(template)
            
            if success:
                # Save templates and update embeddings
                self.template_manager.save_templates()
                await self._update_template_embeddings()
                
                self.logger.info(f"Added new template: {template.id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error adding template from question '{question_text}': {e}", exc_info=True)
            return False

    async def find_template_gaps(self, questions: List[str], 
                               min_cluster_size: int = 3) -> List[Dict[str, Any]]:
        """
        Identify groups of similar questions that don't have matching templates.
        These represent potential gaps in the template database.
        
        Args:
            questions: List of questions to analyze
            min_cluster_size: Minimum number of questions to form a gap cluster
            
        Returns:
            List of gap clusters with suggested template information
        """
        try:
            # Find questions that don't match existing templates
            unmatched_questions = []
            
            for question in questions:
                result = await self.map_question(question)
                if not result["success"] or not result["matched_template"]:
                    unmatched_questions.append(question)
            
            if len(unmatched_questions) < min_cluster_size:
                return []
            
            # Cluster unmatched questions
            clusters = self.similarity_engine.cluster_similar_questions(
                unmatched_questions, similarity_threshold=0.7
            )
            
            # Filter clusters by minimum size
            significant_clusters = [
                cluster for cluster in clusters 
                if len(cluster) >= min_cluster_size
            ]
            
            # Generate gap analysis
            gaps = []
            for i, cluster in enumerate(significant_clusters):
                cluster_questions = [unmatched_questions[idx] for idx in cluster]
                
                # Use the first question as the representative
                representative_question = cluster_questions[0]
                
                gaps.append({
                    "gap_id": f"gap_{i+1}",
                    "representative_question": representative_question,
                    "similar_questions": cluster_questions,
                    "cluster_size": len(cluster_questions),
                    "suggested_template_id": f"template_gap_{i+1}",
                    "suggested_category": "unknown",  # Would need classification
                    "suggested_question_type": "unknown"  # Would need classification
                })
            
            self.logger.info(f"Found {len(gaps)} template gaps from {len(questions)} questions")
            return gaps
            
        except Exception as e:
            self.logger.error(f"Error finding template gaps: {e}", exc_info=True)
            return []

    async def get_mapping_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about question mapping performance.
        
        Returns:
            Dictionary with mapping statistics
        """
        try:
            template_stats = self.template_manager.get_statistics()
            embedding_stats = self.similarity_engine.get_embedding_stats()
            
            return {
                "templates": template_stats,
                "embeddings": embedding_stats,
                "similarity_threshold": self.similarity_threshold,
                "cache_status": {
                    "template_embeddings_cached": self._template_embeddings_cache is not None,
                    "cached_template_count": len(self._template_texts_cache) if self._template_texts_cache else 0
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting mapping statistics: {e}", exc_info=True)
            return {"error": str(e)}

    async def _update_template_embeddings(self) -> None:
        """Update cached template embeddings."""
        try:
            template_texts = self.template_manager.get_template_texts()
            template_ids = self.template_manager.get_template_ids()
            
            if template_texts:
                self._template_embeddings_cache = self.similarity_engine.encode_questions(template_texts)
                self._template_texts_cache = template_texts
                self._template_ids_cache = template_ids
                
                self.logger.debug(f"Updated embeddings for {len(template_texts)} templates")
            else:
                self._template_embeddings_cache = None
                self._template_texts_cache = []
                self._template_ids_cache = []
                
        except Exception as e:
            self.logger.error(f"Error updating template embeddings: {e}", exc_info=True)

    async def _ensure_template_embeddings(self) -> None:
        """Ensure template embeddings are available and up to date."""
        if (self._template_embeddings_cache is None or 
            len(self._template_texts_cache or []) != len(self.template_manager.get_template_texts())):
            await self._update_template_embeddings()

    def set_similarity_threshold(self, threshold: float) -> None:
        """
        Set the similarity threshold for template matching.
        
        Args:
            threshold: New similarity threshold (0.0 to 1.0)
        """
        self.similarity_threshold = max(0.0, min(1.0, threshold))
        self.logger.info(f"Similarity threshold set to {self.similarity_threshold}")

    def get_similarity_threshold(self) -> float:
        """
        Get the current similarity threshold.
        
        Returns:
            Current similarity threshold
        """
        return self.similarity_threshold

