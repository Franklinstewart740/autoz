"""
Extractor Agent
Specialized agent responsible for parsing web pages and extracting survey questions and metadata.
"""

import asyncio
import random
import time
from typing import Any, Dict, List, Optional
from bs4 import BeautifulSoup

from .base_agent import BaseAgent, AgentTask


class ExtractorAgent(BaseAgent):
    """
    Extractor Agent handles parsing web pages and extracting structured data.
    Specialized in:
    - Identifying survey questions and answer options
    - Extracting survey metadata (length, reward, time)
    - Handling dynamic content and SPAs
    - Classifying question types
    """

    def __init__(self, agent_id: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(agent_id, "extractor", config)
        
        self.extraction_rules = config.get("extraction_rules", {})
        self.question_types = [
            "multiple_choice", "single_choice", "open_ended", "likert_scale",
            "ranking", "dropdown", "checkbox", "numeric", "date"
        ]
        self.last_extracted_data = {}

    async def initialize(self) -> bool:
        """Initialize the Extractor Agent."""
        self.logger.info("Initializing Extractor Agent")
        # No external resources to initialize for now
        return True

    async def cleanup(self) -> None:
        """Clean up Extractor Agent resources."""
        self.logger.info("Extractor Agent cleanup completed")
        pass

    async def process_task(self, task: AgentTask) -> Dict[str, Any]:
        """
        Process extraction-related tasks.
        
        Args:
            task: The extraction task to process
            
        Returns:
            Dictionary containing the task result
        """
        task_type = task.task_type
        payload = task.payload
        
        self.logger.info(f"Processing extraction task: {task_type}")
        
        try:
            if task_type == "extract_survey_questions":
                return await self._extract_survey_questions(payload)
            
            elif task_type == "extract_survey_metadata":
                return await self._extract_survey_metadata(payload)
            
            elif task_type == "classify_question_type":
                return await self._classify_question_type(payload)
            
            else:
                raise ValueError(f"Unknown extraction task type: {task_type}")
                
        except Exception as e:
            self.logger.error(f"Error processing extraction task {task_type}: {e}", exc_info=True)
            raise

    async def _extract_survey_questions(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Extract survey questions and options from page content."""
        html_content = payload["html_content"]
        current_url = payload["current_url"]
        
        self.logger.debug(f"Extracting questions from {current_url}")
        
        soup = BeautifulSoup(html_content, "html.parser")
        questions = []
        
        # Example: Look for common question patterns
        # This will need to be refined with more platform-specific rules
        question_elements = soup.find_all(["h1", "h2", "h3", "p", "div", "span"], class_=[lambda x: x and ("question" in x or "survey-q" in x)])
        
        for i, q_elem in enumerate(question_elements):
            question_text = q_elem.get_text(strip=True)
            if not question_text:
                continue
            
            question_data = {
                "id": f"q_{i}_{random.randint(1000, 9999)}",
                "text": question_text,
                "type": "unknown", # Will be classified later
                "options": [],
                "required": False,
                "element_selector": self._get_selector(q_elem)
            }
            
            # Attempt to find options associated with the question
            # This is a very basic approach and needs significant improvement for real-world scenarios
            options_container = q_elem.find_next_sibling(["div", "ul", "ol"])
            if options_container:
                option_elements = options_container.find_all(["li", "div", "label"], class_=[lambda x: x and ("option" in x or "answer" in x)])
                for j, opt_elem in enumerate(option_elements):
                    option_text = opt_elem.get_text(strip=True)
                    if option_text:
                        question_data["options"].append({
                            "id": f"opt_{j}_{random.randint(1000, 9999)}",
                            "text": option_text,
                            "value": opt_elem.get("value") or opt_elem.get("data-value") or option_text,
                            "selector": self._get_selector(opt_elem)
                        })
            questions.append(question_data)
            
        self.last_extracted_data["questions"] = questions
        return {
            "success": True,
            "questions": questions,
            "count": len(questions)
        }

    async def _extract_survey_metadata(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Extract survey metadata from page content."""
        html_content = payload["html_content"]
        current_url = payload["current_url"]
        
        self.logger.debug(f"Extracting metadata from {current_url}")
        
        soup = BeautifulSoup(html_content, "html.parser")
        metadata = {}
        
        # Example: Look for common metadata patterns
        # This will need to be refined with more platform-specific rules
        reward_elem = soup.find(class_=lambda x: x and ("reward" in x or "points" in x))
        if reward_elem: metadata["reward"] = reward_elem.get_text(strip=True)
        
        time_elem = soup.find(class_=lambda x: x and ("time" in x or "duration" in x))
        if time_elem: metadata["estimated_time"] = time_elem.get_text(strip=True)
        
        length_elem = soup.find(class_=lambda x: x and ("length" in x or "progress" in x))
        if length_elem: metadata["progress"] = length_elem.get_text(strip=True)
        
        self.last_extracted_data["metadata"] = metadata
        return {
            "success": True,
            "metadata": metadata
        }

    async def _classify_question_type(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Classify the type of a given survey question."""
        question_text = payload["question_text"]
        options = payload.get("options", [])
        
        # Simple heuristic-based classification for now
        if not options and ("open ended" in question_text.lower() or "describe" in question_text.lower()):
            question_type = "open_ended"
        elif len(options) > 0:
            if any("select all" in question_text.lower() for _ in options):
                question_type = "multiple_choice" # Checkbox
            else:
                question_type = "single_choice" # Radio button or dropdown
        elif "rate" in question_text.lower() or "scale" in question_text.lower() or "agree" in question_text.lower():
            question_type = "likert_scale"
        elif "rank" in question_text.lower():
            question_type = "ranking"
        else:
            question_type = "open_ended" # Default to open-ended if unsure
            
        return {
            "success": True,
            "question_text": question_text,
            "classified_type": question_type
        }

    def _get_selector(self, element) -> str:
        """
        Generates a CSS selector for a given BeautifulSoup element.
        This is a basic implementation and might need more robustness.
        """
        path = []
        for parent in element.parents:
            if parent.name:
                siblings = parent.find_all(element.name, recursive=False)
                if len(siblings) > 1:
                    idx = siblings.index(element) + 1
                    path.append(f"{element.name}:nth-of-type({idx})")
                else:
                    path.append(element.name)
                element = parent
            else:
                break
        return " > ".join(reversed(path))

