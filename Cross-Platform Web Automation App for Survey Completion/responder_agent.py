"""
Responder Agent
Specialized agent responsible for generating AI-powered responses to survey questions.
"""

import asyncio
import random
import time
from typing import Any, Dict, List, Optional
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util

from .base_agent import BaseAgent, AgentTask


class ResponderAgent(BaseAgent):
    """
    Responder Agent generates contextually appropriate and human-like responses.
    Specialized in:
    - Integrating with various LLMs (OpenAI, HuggingFace, Ollama, Groq)
    - Generating responses for different question types
    - Leveraging semantic anchoring for response relevance
    - Adapting responses based on persona and survey context
    """

    def __init__(self, agent_id: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(agent_id, "responder", config)
        
        self.openai_client = None
        self.embedding_model = None
        self.llm_backend = config.get("llm_backend", "openai") # openai, huggingface, ollama, groq
        self.persona = config.get("persona", "default")
        self.personas = {
            "default": "You are a helpful assistant.",
            "young_professional": "You are a young professional, tech-savvy, career-focused, and live an urban lifestyle. Your responses are articulate and forward-thinking.",
            "family_oriented": "You are a family-oriented individual, budget-conscious, and live in a suburban area. Your responses reflect practical concerns and family values.",
            "retiree": "You are a retiree, experienced, value traditional approaches, and enjoy leisure time. Your responses are thoughtful and consider long-term implications.",
            "student": "You are a student, budget-conscious, social, and aware of current trends. Your responses are concise and reflect a modern perspective."
        }
        
        # Semantic anchoring data (example structure)
        self.known_questions = [
            {"question": "What is your age?", "template": "age_question", "response_strategy": "numeric"},
            {"question": "How satisfied are you with our service?", "template": "satisfaction_likert", "response_strategy": "likert"},
            {"question": "Describe your experience.", "template": "open_ended_experience", "response_strategy": "open_ended"},
        ]
        self.known_question_embeddings = []
        
        self.response_history = [] # Short-term memory for coherent responses

    async def initialize(self) -> bool:
        """Initialize the Responder Agent with LLM clients and embedding models."""
        self.logger.info("Initializing Responder Agent")
        try:
            # Initialize OpenAI client if selected
            if self.llm_backend == "openai":
                self.openai_client = OpenAI()
            
            # Initialize SentenceTransformer for semantic anchoring
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            self._precompute_question_embeddings()
            
            self.logger.info("Responder Agent initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Responder Agent: {e}", exc_info=True)
            return False

    async def cleanup(self) -> None:
        """Clean up Responder Agent resources."""
        self.logger.info("Responder Agent cleanup completed")
        # No specific cleanup needed for these clients
        pass

    async def process_task(self, task: AgentTask) -> Dict[str, Any]:
        """
        Process response generation tasks.
        
        Args:
            task: The response generation task to process
            
        Returns:
            Dictionary containing the task result
        """
        task_type = task.task_type
        payload = task.payload
        
        self.logger.info(f"Processing response generation task: {task_type}")
        
        try:
            if task_type == "generate_response":
                return await self._generate_response(payload)
            
            else:
                raise ValueError(f"Unknown response generation task type: {task_type}")
                
        except Exception as e:
            self.logger.error(f"Error processing response generation task {task_type}: {e}", exc_info=True)
            raise

    async def _generate_response(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a response for a given survey question."""
        question = payload["question"]
        question_text = question["text"]
        question_type = question["type"]
        options = question.get("options", [])
        
        self.logger.debug(f"Generating response for question: {question_text} ({question_type})")
        
        # Retrieve persona instruction
        persona_instruction = self.personas.get(self.persona, self.personas["default"])
        
        # Semantic anchoring to find best response strategy
        anchored_strategy = self._get_anchored_response_strategy(question_text)
        
        # Build prompt based on question type, persona, and context
        prompt = self._build_prompt(question_text, question_type, options, persona_instruction, anchored_strategy)
        
        response_text = ""
        confidence = 0.0
        
        try:
            if self.llm_backend == "openai" and self.openai_client:
                completion = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": persona_instruction},
                        {"role": "user", "content": prompt}
                    ]
                )
                response_text = completion.choices[0].message.content.strip()
                confidence = 0.8 # Placeholder for actual model confidence
            
            # TODO: Add logic for HuggingFace, Ollama, Groq
            else:
                response_text = f"[Placeholder response for {question_type}: {question_text}]"
                confidence = 0.5
                
            # Process response based on question type (e.g., select option for multiple choice)
            processed_response = self._process_generated_response(response_text, question_type, options)
            
            # Update response history for short-term memory
            self.response_history.append({
                "question": question_text,
                "response": processed_response,
                "timestamp": time.time()
            })
            
            return {
                "success": True,
                "question_id": question["id"],
                "response": processed_response,
                "confidence": confidence,
                "original_llm_response": response_text
            }
            
        except Exception as e:
            self.logger.error(f"Error generating response for question 
{question_text}: {e}", exc_info=True)
            return {
                "success": False,
                "question_id": question["id"],
                "error": str(e),
                "confidence": 0.0
            }

    def _precompute_question_embeddings(self) -> None:
        """
        Precompute embeddings for known questions for semantic anchoring.
        This should be done once during initialization.
        """
        if self.embedding_model and self.known_questions:
            questions_list = [q["question"] for q in self.known_questions]
            self.known_question_embeddings = self.embedding_model.encode(questions_list, convert_to_tensor=True)
            self.logger.info(f"Precomputed embeddings for {len(self.known_questions)} known questions.")

    def _get_anchored_response_strategy(self, current_question_text: str) -> Optional[Dict[str, Any]]:
        """
        Uses semantic similarity to find a matching known question and its strategy.
        """
        if not self.embedding_model or not self.known_question_embeddings:
            return None
            
        current_question_embedding = self.embedding_model.encode(current_question_text, convert_to_tensor=True)
        
        # Compute cosine similarity between current question and known questions
        cosine_scores = util.cos_sim(current_question_embedding, self.known_question_embeddings)[0]
        
        # Find the most similar known question
        best_match_idx = cosine_scores.argmax().item()
        best_score = cosine_scores[best_match_idx].item()
        
        if best_score > self.config.get("semantic_similarity_threshold", 0.7):
            self.logger.debug(f"Question '{current_question_text}' semantically anchored to '{self.known_questions[best_match_idx]['question']}' with score {best_score:.2f}")
            return self.known_questions[best_match_idx]
        
        return None

    def _build_prompt(self, question_text: str, question_type: str, options: List[Dict[str, Any]], persona_instruction: str, anchored_strategy: Optional[Dict[str, Any]]) -> str:
        """
        Builds the LLM prompt based on question details, persona, and context.
        """
        prompt_parts = [
            f"You are acting as a survey participant with the following persona: {persona_instruction}"
        ]
        
        # Add short-term memory to prompt
        if self.response_history:
            prompt_parts.append("Here is your recent survey response history for context:")
            for entry in self.response_history[-3:]: # Include last 3 responses
                prompt_parts.append(f"- Question: {entry['question']}\n  Your previous answer: {entry['response']}")
        
        prompt_parts.append(f"Please answer the following survey question. The question type is {question_type}.")
        
        if anchored_strategy:
            prompt_parts.append(f"Based on semantic analysis, this question is similar to a known template '{anchored_strategy['template']}'. Consider its response strategy: {anchored_strategy['response_strategy']}.")
            
        prompt_parts.append(f"Question: {question_text}")
        
        if options:
            prompt_parts.append("Available options:")
            for i, opt in enumerate(options):
                prompt_parts.append(f"{i+1}. {opt['text']}")
            prompt_parts.append("Your answer should be the number corresponding to the best option, or a comma-separated list of numbers for multiple selections.")
        elif question_type == "likert_scale":
            prompt_parts.append("Please provide a rating on a scale, or choose from options like 'Strongly Agree', 'Agree', 'Neutral', 'Disagree', 'Strongly Disagree'.")
        elif question_type == "ranking":
            prompt_parts.append("Please rank the options provided (e.g., '1. Option A, 2. Option B').")
        
        prompt_parts.append("Your response:")
        
        return "\n".join(prompt_parts)

    def _process_generated_response(self, llm_response: str, question_type: str, options: List[Dict[str, Any]]) -> Any:
        """
        Processes the raw LLM response to fit the expected format for each question type.
        """
        if question_type in ["multiple_choice", "single_choice"] and options:
            # Try to match the LLM's response to an option ID or text
            llm_response_lower = llm_response.lower()
            
            # Attempt to match by number if options are numbered in prompt
            if llm_response.isdigit() and 1 <= int(llm_response) <= len(options):
                return options[int(llm_response) - 1]["value"]
            
            # Attempt to match by text
            for opt in options:
                if opt["text"].lower() == llm_response_lower:
                    return opt["value"]
            
            # Fallback: if no direct match, try to find closest match or return first option
            # This needs more sophisticated logic (e.g., fuzzy matching, embedding similarity)
            if options:
                self.logger.warning(f"LLM response '{llm_response}' did not directly match options. Returning first option as fallback.")
                return options[0]["value"]
            
            return llm_response # If no options, return as is
            
        # For open-ended, likert, ranking, etc., return the response as is
        return llm_response

