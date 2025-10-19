"""
Survey Type Classifier
Pre-processing module that classifies survey types to enable tailored response strategies.
"""

import logging
import pickle
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd


class SurveyClassifier:
    """
    Classifies surveys into predefined categories to enable tailored response strategies.
    
    Survey Categories:
    - opinion_poll: General opinion surveys about current events, politics, etc.
    - product_feedback: Product reviews, satisfaction surveys, user experience
    - demographic_profiling: Personal information, lifestyle, preferences
    - market_research: Brand awareness, purchasing behavior, market trends
    - academic_research: Educational surveys, research studies
    - customer_service: Support satisfaction, service quality assessment
    """

    def __init__(self, model_path: Optional[str] = None):
        self.logger = logging.getLogger("survey_classifier")
        
        # Survey categories
        self.categories = [
            "opinion_poll",
            "product_feedback", 
            "demographic_profiling",
            "market_research",
            "academic_research",
            "customer_service"
        ]
        
        # Classification pipeline
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                lowercase=True
            )),
            ('classifier', MultinomialNB(alpha=0.1))
        ])
        
        self.is_trained = False
        self.model_path = model_path
        
        # Load pre-trained model if available
        if model_path:
            self.load_model(model_path)

    def extract_features(self, survey_metadata: Dict[str, Any]) -> str:
        """
        Extract text features from survey metadata for classification.
        
        Args:
            survey_metadata: Dictionary containing survey information
            
        Returns:
            Combined text string for classification
        """
        features = []
        
        # Title and description
        if 'title' in survey_metadata:
            features.append(survey_metadata['title'])
        if 'description' in survey_metadata:
            features.append(survey_metadata['description'])
        
        # Question samples (first few questions)
        if 'questions' in survey_metadata:
            questions = survey_metadata['questions'][:3]  # Use first 3 questions
            for q in questions:
                if isinstance(q, dict) and 'text' in q:
                    features.append(q['text'])
                elif isinstance(q, str):
                    features.append(q)
        
        # Platform and source information
        if 'platform' in survey_metadata:
            features.append(f"platform_{survey_metadata['platform']}")
        
        # Reward and time information (as text features)
        if 'reward' in survey_metadata:
            features.append(f"reward_{survey_metadata['reward']}")
        if 'estimated_time' in survey_metadata:
            features.append(f"time_{survey_metadata['estimated_time']}")
        
        return " ".join(features)

    def classify_survey(self, survey_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify a survey based on its metadata.
        
        Args:
            survey_metadata: Dictionary containing survey information
            
        Returns:
            Dictionary with classification results
        """
        if not self.is_trained:
            self.logger.warning("Classifier not trained. Using heuristic classification.")
            return self._heuristic_classification(survey_metadata)
        
        try:
            # Extract features
            text_features = self.extract_features(survey_metadata)
            
            # Predict category
            predicted_category = self.pipeline.predict([text_features])[0]
            
            # Get prediction probabilities
            probabilities = self.pipeline.predict_proba([text_features])[0]
            confidence = max(probabilities)
            
            # Create probability dictionary
            category_probabilities = dict(zip(self.categories, probabilities))
            
            self.logger.info(f"Classified survey as '{predicted_category}' with confidence {confidence:.3f}")
            
            return {
                "category": predicted_category,
                "confidence": confidence,
                "probabilities": category_probabilities,
                "strategy": self._get_response_strategy(predicted_category)
            }
            
        except Exception as e:
            self.logger.error(f"Error classifying survey: {e}", exc_info=True)
            return self._heuristic_classification(survey_metadata)

    def _heuristic_classification(self, survey_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback heuristic-based classification when ML model is not available.
        
        Args:
            survey_metadata: Dictionary containing survey information
            
        Returns:
            Dictionary with classification results
        """
        text_content = self.extract_features(survey_metadata).lower()
        
        # Simple keyword-based classification
        if any(keyword in text_content for keyword in ['product', 'service', 'experience', 'satisfaction', 'review']):
            category = "product_feedback"
        elif any(keyword in text_content for keyword in ['age', 'gender', 'income', 'location', 'demographic']):
            category = "demographic_profiling"
        elif any(keyword in text_content for keyword in ['brand', 'purchase', 'buy', 'market', 'advertising']):
            category = "market_research"
        elif any(keyword in text_content for keyword in ['opinion', 'political', 'vote', 'policy', 'government']):
            category = "opinion_poll"
        elif any(keyword in text_content for keyword in ['research', 'study', 'academic', 'university']):
            category = "academic_research"
        elif any(keyword in text_content for keyword in ['support', 'help', 'customer service', 'assistance']):
            category = "customer_service"
        else:
            category = "market_research"  # Default category
        
        self.logger.info(f"Heuristically classified survey as '{category}'")
        
        return {
            "category": category,
            "confidence": 0.6,  # Lower confidence for heuristic classification
            "probabilities": {cat: 0.6 if cat == category else 0.08 for cat in self.categories},
            "strategy": self._get_response_strategy(category)
        }

    def _get_response_strategy(self, category: str) -> Dict[str, Any]:
        """
        Get the response strategy configuration for a given survey category.
        
        Args:
            category: Survey category
            
        Returns:
            Dictionary with strategy configuration
        """
        strategies = {
            "opinion_poll": {
                "persona_weight": 0.8,  # High persona influence
                "consistency_weight": 0.7,
                "response_length": "medium",
                "preferred_models": ["gpt-3.5-turbo", "claude"],
                "special_instructions": "Express clear opinions while remaining respectful"
            },
            "product_feedback": {
                "persona_weight": 0.9,
                "consistency_weight": 0.8,
                "response_length": "detailed",
                "preferred_models": ["gpt-4", "gpt-3.5-turbo"],
                "special_instructions": "Provide specific, constructive feedback based on realistic usage scenarios"
            },
            "demographic_profiling": {
                "persona_weight": 1.0,  # Maximum persona influence
                "consistency_weight": 1.0,  # Must be consistent
                "response_length": "short",
                "preferred_models": ["gpt-3.5-turbo"],
                "special_instructions": "Maintain strict consistency with established persona demographics"
            },
            "market_research": {
                "persona_weight": 0.7,
                "consistency_weight": 0.6,
                "response_length": "medium",
                "preferred_models": ["gpt-3.5-turbo", "claude"],
                "special_instructions": "Reflect realistic consumer behavior and preferences"
            },
            "academic_research": {
                "persona_weight": 0.6,
                "consistency_weight": 0.9,
                "response_length": "detailed",
                "preferred_models": ["gpt-4", "claude"],
                "special_instructions": "Provide thoughtful, well-reasoned responses suitable for research"
            },
            "customer_service": {
                "persona_weight": 0.8,
                "consistency_weight": 0.7,
                "response_length": "medium",
                "preferred_models": ["gpt-3.5-turbo"],
                "special_instructions": "Focus on service experience and satisfaction levels"
            }
        }
        
        return strategies.get(category, strategies["market_research"])

    def train_model(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Train the classification model on provided training data.
        
        Args:
            training_data: List of dictionaries with 'metadata' and 'category' keys
            
        Returns:
            Dictionary with training results
        """
        if len(training_data) < 10:
            self.logger.warning("Insufficient training data. Need at least 10 samples.")
            return {"success": False, "error": "Insufficient training data"}
        
        try:
            # Prepare training data
            X = []
            y = []
            
            for sample in training_data:
                text_features = self.extract_features(sample['metadata'])
                X.append(text_features)
                y.append(sample['category'])
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train model
            self.pipeline.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            self.is_trained = True
            
            self.logger.info(f"Model trained successfully. Accuracy: {accuracy:.3f}")
            
            return {
                "success": True,
                "accuracy": accuracy,
                "classification_report": report,
                "training_samples": len(X_train),
                "test_samples": len(X_test)
            }
            
        except Exception as e:
            self.logger.error(f"Error training model: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def save_model(self, path: str) -> bool:
        """
        Save the trained model to disk.
        
        Args:
            path: File path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_trained:
            self.logger.error("Cannot save untrained model")
            return False
        
        try:
            with open(path, 'wb') as f:
                pickle.dump({
                    'pipeline': self.pipeline,
                    'categories': self.categories,
                    'is_trained': self.is_trained
                }, f)
            
            self.logger.info(f"Model saved to {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}", exc_info=True)
            return False

    def load_model(self, path: str) -> bool:
        """
        Load a trained model from disk.
        
        Args:
            path: File path to load the model from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.pipeline = model_data['pipeline']
            self.categories = model_data['categories']
            self.is_trained = model_data['is_trained']
            
            self.logger.info(f"Model loaded from {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}", exc_info=True)
            return False

    def get_training_data_template(self) -> List[Dict[str, Any]]:
        """
        Get a template for training data format.
        
        Returns:
            List of example training data entries
        """
        return [
            {
                "metadata": {
                    "title": "Product Satisfaction Survey",
                    "description": "Tell us about your experience with our product",
                    "questions": [
                        {"text": "How satisfied are you with the product quality?"},
                        {"text": "Would you recommend this product to others?"}
                    ],
                    "platform": "swagbucks",
                    "reward": "50 points",
                    "estimated_time": "5 minutes"
                },
                "category": "product_feedback"
            },
            {
                "metadata": {
                    "title": "Political Opinion Poll",
                    "description": "Share your views on current political issues",
                    "questions": [
                        {"text": "What is your opinion on the current government policies?"},
                        {"text": "Which political party do you support?"}
                    ],
                    "platform": "inboxdollars",
                    "reward": "$0.75",
                    "estimated_time": "8 minutes"
                },
                "category": "opinion_poll"
            }
        ]

