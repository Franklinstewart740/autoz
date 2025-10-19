"""
Question Type Classifier
Classifies individual survey questions into specific types for targeted response generation.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import pickle


class QuestionClassifier:
    """
    Classifies individual survey questions into specific types.
    
    Question Types:
    - multiple_choice: Questions with multiple options, select one
    - multiple_select: Questions with multiple options, select multiple
    - likert_scale: Rating scale questions (1-5, strongly agree/disagree, etc.)
    - open_ended: Free text response questions
    - numeric: Questions requiring numeric input
    - date: Questions asking for dates
    - ranking: Questions asking to rank options in order
    - yes_no: Simple yes/no questions
    - dropdown: Single selection from dropdown menu
    - slider: Continuous scale questions
    """

    def __init__(self, model_path: Optional[str] = None):
        self.logger = logging.getLogger("question_classifier")
        
        # Question types
        self.question_types = [
            "multiple_choice",
            "multiple_select", 
            "likert_scale",
            "open_ended",
            "numeric",
            "date",
            "ranking",
            "yes_no",
            "dropdown",
            "slider"
        ]
        
        # Classification pipeline
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=500,
                stop_words='english',
                ngram_range=(1, 3),
                lowercase=True
            )),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10
            ))
        ])
        
        self.is_trained = False
        self.model_path = model_path
        
        # Keyword patterns for heuristic classification
        self.keyword_patterns = {
            "likert_scale": [
                r"strongly\s+(agree|disagree)",
                r"rate.*scale",
                r"how\s+(satisfied|likely|important)",
                r"\d+\s*-\s*\d+\s*scale",
                r"(very|extremely|somewhat|not at all)",
                r"(excellent|good|fair|poor)"
            ],
            "yes_no": [
                r"yes\s*or\s*no",
                r"true\s*or\s*false",
                r"do\s+you\s+",
                r"have\s+you\s+",
                r"are\s+you\s+",
                r"would\s+you\s+"
            ],
            "numeric": [
                r"how\s+many",
                r"what\s+is\s+your\s+age",
                r"enter\s+a\s+number",
                r"numeric\s+value",
                r"years?\s+old",
                r"income|salary|wage"
            ],
            "date": [
                r"date\s+of\s+birth",
                r"when\s+did\s+you",
                r"what\s+date",
                r"mm/dd/yyyy",
                r"birthday|birthdate"
            ],
            "ranking": [
                r"rank\s+the\s+following",
                r"order\s+of\s+preference",
                r"prioritize",
                r"arrange\s+in\s+order"
            ],
            "open_ended": [
                r"describe\s+",
                r"explain\s+",
                r"tell\s+us\s+about",
                r"in\s+your\s+own\s+words",
                r"comments?",
                r"feedback"
            ]
        }
        
        # Load pre-trained model if available
        if model_path:
            self.load_model(model_path)

    def extract_features(self, question_data: Dict[str, Any]) -> str:
        """
        Extract text features from question data for classification.
        
        Args:
            question_data: Dictionary containing question information
            
        Returns:
            Combined text string for classification
        """
        features = []
        
        # Question text
        if 'text' in question_data:
            features.append(question_data['text'])
        
        # Options information
        if 'options' in question_data and question_data['options']:
            options = question_data['options']
            features.append(f"has_options_{len(options)}_options")
            
            # Sample option texts
            for i, opt in enumerate(options[:3]):  # First 3 options
                if isinstance(opt, dict) and 'text' in opt:
                    features.append(f"option_{i}_{opt['text']}")
                elif isinstance(opt, str):
                    features.append(f"option_{i}_{opt}")
        else:
            features.append("no_options")
        
        # HTML element type hints
        if 'element_type' in question_data:
            features.append(f"element_{question_data['element_type']}")
        
        # Required field indicator
        if question_data.get('required', False):
            features.append("required_field")
        
        return " ".join(features)

    def classify_question(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify a question based on its data.
        
        Args:
            question_data: Dictionary containing question information
            
        Returns:
            Dictionary with classification results
        """
        if not self.is_trained:
            self.logger.warning("Classifier not trained. Using heuristic classification.")
            return self._heuristic_classification(question_data)
        
        try:
            # Extract features
            text_features = self.extract_features(question_data)
            
            # Predict question type
            predicted_type = self.pipeline.predict([text_features])[0]
            
            # Get prediction probabilities
            probabilities = self.pipeline.predict_proba([text_features])[0]
            confidence = max(probabilities)
            
            # Create probability dictionary
            type_probabilities = dict(zip(self.question_types, probabilities))
            
            self.logger.debug(f"Classified question as '{predicted_type}' with confidence {confidence:.3f}")
            
            return {
                "question_type": predicted_type,
                "confidence": confidence,
                "probabilities": type_probabilities,
                "response_strategy": self._get_response_strategy(predicted_type)
            }
            
        except Exception as e:
            self.logger.error(f"Error classifying question: {e}", exc_info=True)
            return self._heuristic_classification(question_data)

    def _heuristic_classification(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback heuristic-based classification when ML model is not available.
        
        Args:
            question_data: Dictionary containing question information
            
        Returns:
            Dictionary with classification results
        """
        question_text = question_data.get('text', '').lower()
        options = question_data.get('options', [])
        
        # Check for options first
        if options:
            # Check if it's a Likert scale based on option content
            option_texts = [opt.get('text', '') if isinstance(opt, dict) else str(opt) for opt in options]
            option_text_combined = ' '.join(option_texts).lower()
            
            if any(keyword in option_text_combined for keyword in ['strongly agree', 'agree', 'disagree', 'excellent', 'good', 'poor']):
                question_type = "likert_scale"
            elif len(options) == 2 and any(keyword in option_text_combined for keyword in ['yes', 'no', 'true', 'false']):
                question_type = "yes_no"
            elif 'select all' in question_text or 'check all' in question_text:
                question_type = "multiple_select"
            elif len(options) > 5:
                question_type = "dropdown"
            else:
                question_type = "multiple_choice"
        else:
            # No options - classify based on question text patterns
            question_type = "open_ended"  # Default
            
            for qtype, patterns in self.keyword_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, question_text, re.IGNORECASE):
                        question_type = qtype
                        break
                if question_type != "open_ended":
                    break
        
        self.logger.debug(f"Heuristically classified question as '{question_type}'")
        
        return {
            "question_type": question_type,
            "confidence": 0.7,  # Moderate confidence for heuristic classification
            "probabilities": {qtype: 0.7 if qtype == question_type else 0.03 for qtype in self.question_types},
            "response_strategy": self._get_response_strategy(question_type)
        }

    def _get_response_strategy(self, question_type: str) -> Dict[str, Any]:
        """
        Get the response strategy configuration for a given question type.
        
        Args:
            question_type: Question type
            
        Returns:
            Dictionary with strategy configuration
        """
        strategies = {
            "multiple_choice": {
                "selection_method": "single",
                "randomization": True,
                "persona_influence": 0.8,
                "response_time_range": (2, 5)  # seconds
            },
            "multiple_select": {
                "selection_method": "multiple",
                "max_selections": 3,
                "randomization": True,
                "persona_influence": 0.7,
                "response_time_range": (3, 8)
            },
            "likert_scale": {
                "selection_method": "scale",
                "avoid_extremes": True,
                "persona_influence": 0.9,
                "response_time_range": (2, 4)
            },
            "open_ended": {
                "selection_method": "text_generation",
                "min_length": 10,
                "max_length": 200,
                "persona_influence": 1.0,
                "response_time_range": (5, 15)
            },
            "numeric": {
                "selection_method": "numeric_generation",
                "persona_influence": 0.9,
                "validation_required": True,
                "response_time_range": (2, 5)
            },
            "date": {
                "selection_method": "date_generation",
                "persona_influence": 1.0,
                "validation_required": True,
                "response_time_range": (3, 7)
            },
            "ranking": {
                "selection_method": "ranking",
                "persona_influence": 0.8,
                "response_time_range": (5, 12)
            },
            "yes_no": {
                "selection_method": "binary",
                "persona_influence": 0.8,
                "response_time_range": (1, 3)
            },
            "dropdown": {
                "selection_method": "single",
                "randomization": True,
                "persona_influence": 0.7,
                "response_time_range": (2, 6)
            },
            "slider": {
                "selection_method": "continuous",
                "avoid_extremes": True,
                "persona_influence": 0.8,
                "response_time_range": (2, 5)
            }
        }
        
        return strategies.get(question_type, strategies["open_ended"])

    def train_model(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Train the classification model on provided training data.
        
        Args:
            training_data: List of dictionaries with 'question_data' and 'question_type' keys
            
        Returns:
            Dictionary with training results
        """
        if len(training_data) < 20:
            self.logger.warning("Insufficient training data. Need at least 20 samples.")
            return {"success": False, "error": "Insufficient training data"}
        
        try:
            # Prepare training data
            X = []
            y = []
            
            for sample in training_data:
                text_features = self.extract_features(sample['question_data'])
                X.append(text_features)
                y.append(sample['question_type'])
            
            # Train model
            self.pipeline.fit(X, y)
            
            # Simple evaluation (in practice, use train/test split)
            y_pred = self.pipeline.predict(X)
            accuracy = sum(1 for i, j in zip(y, y_pred) if i == j) / len(y)
            
            self.is_trained = True
            
            self.logger.info(f"Question classifier trained successfully. Accuracy: {accuracy:.3f}")
            
            return {
                "success": True,
                "accuracy": accuracy,
                "training_samples": len(X)
            }
            
        except Exception as e:
            self.logger.error(f"Error training question classifier: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def save_model(self, path: str) -> bool:
        """Save the trained model to disk."""
        if not self.is_trained:
            self.logger.error("Cannot save untrained model")
            return False
        
        try:
            with open(path, 'wb') as f:
                pickle.dump({
                    'pipeline': self.pipeline,
                    'question_types': self.question_types,
                    'is_trained': self.is_trained
                }, f)
            
            self.logger.info(f"Question classifier saved to {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving question classifier: {e}", exc_info=True)
            return False

    def load_model(self, path: str) -> bool:
        """Load a trained model from disk."""
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.pipeline = model_data['pipeline']
            self.question_types = model_data['question_types']
            self.is_trained = model_data['is_trained']
            
            self.logger.info(f"Question classifier loaded from {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading question classifier: {e}", exc_info=True)
            return False

