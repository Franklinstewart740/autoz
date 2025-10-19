"""
AI Response Generator Module
Generates intelligent responses to survey questions using AI models.
"""

import asyncio
import logging
import random
import re
from typing import Dict, Any, List, Optional, Union
import openai
import os
from .survey_extractor import SurveyQuestion, QuestionType

logger = logging.getLogger(__name__)

class AIResponseGenerator:
    """Generates AI-powered responses to survey questions."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = None
        
        if self.api_key:
            self.client = openai.OpenAI(api_key=self.api_key)
        else:
            logger.warning("No OpenAI API key provided. AI responses will be simulated.")
        
        # Response templates for different question types
        self.response_templates = {
            QuestionType.MULTIPLE_CHOICE: self._generate_multiple_choice_response,
            QuestionType.SINGLE_CHOICE: self._generate_single_choice_response,
            QuestionType.TEXT_INPUT: self._generate_text_response,
            QuestionType.TEXTAREA: self._generate_long_text_response,
            QuestionType.RATING_SCALE: self._generate_rating_response,
            QuestionType.LIKERT_SCALE: self._generate_likert_response,
            QuestionType.RANKING: self._generate_ranking_response,
            QuestionType.CHECKBOX: self._generate_checkbox_response,
            QuestionType.DROPDOWN: self._generate_dropdown_response,
            QuestionType.SLIDER: self._generate_slider_response,
            QuestionType.DATE_INPUT: self._generate_date_response,
            QuestionType.NUMBER_INPUT: self._generate_number_response
        }
        
        # Persona profiles for varied responses
        self.personas = [
            {
                "name": "young_professional",
                "age_range": "25-35",
                "characteristics": "tech-savvy, career-focused, urban lifestyle",
                "preferences": "convenience, efficiency, modern brands"
            },
            {
                "name": "family_oriented",
                "age_range": "30-45",
                "characteristics": "family-focused, budget-conscious, suburban",
                "preferences": "value for money, family-friendly products, safety"
            },
            {
                "name": "retiree",
                "age_range": "60+",
                "characteristics": "experienced, traditional values, leisure time",
                "preferences": "quality, reliability, customer service"
            },
            {
                "name": "student",
                "age_range": "18-25",
                "characteristics": "budget-conscious, social, trend-aware",
                "preferences": "affordable options, social media, trendy brands"
            }
        ]
        
        self.current_persona = random.choice(self.personas)
    
    async def generate_response(self, question: SurveyQuestion, context: Dict[str, Any] = None) -> Optional[str]:
        """Generate a response for a survey question.
        
        Args:
            question: The survey question to answer
            context: Additional context about the survey or previous answers
            
        Returns:
            Generated response or None if failed
        """
        try:
            if question.question_type in self.response_templates:
                generator_func = self.response_templates[question.question_type]
                response = await generator_func(question, context or {})
                
                logger.info(f"Generated response for question type {question.question_type.value}: {response}")
                return response
            else:
                logger.warning(f"No response generator for question type: {question.question_type}")
                return await self._generate_fallback_response(question)
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return await self._generate_fallback_response(question)
    
    async def _generate_multiple_choice_response(self, question: SurveyQuestion, context: Dict[str, Any]) -> str:
        """Generate response for multiple choice questions."""
        if not question.options:
            return ""
        
        if self.client:
            try:
                prompt = self._build_ai_prompt(question, context)
                response = await self._call_openai(prompt)
                
                # Extract selected options from AI response
                selected = self._extract_options_from_ai_response(response, question.options)
                if selected:
                    return ",".join(selected)
            except Exception as e:
                logger.warning(f"AI generation failed, using fallback: {e}")
        
        # Fallback: Select 1-3 random options
        num_selections = min(random.randint(1, 3), len(question.options))
        selected = random.sample(question.options, num_selections)
        return ",".join(selected)
    
    async def _generate_single_choice_response(self, question: SurveyQuestion, context: Dict[str, Any]) -> str:
        """Generate response for single choice questions."""
        if not question.options:
            return ""
        
        if self.client:
            try:
                prompt = self._build_ai_prompt(question, context)
                response = await self._call_openai(prompt)
                
                # Extract single option from AI response
                selected = self._extract_options_from_ai_response(response, question.options)
                if selected:
                    return selected[0]
            except Exception as e:
                logger.warning(f"AI generation failed, using fallback: {e}")
        
        # Fallback: Select random option with weighted preference
        return self._weighted_random_choice(question.options)
    
    async def _generate_text_response(self, question: SurveyQuestion, context: Dict[str, Any]) -> str:
        """Generate response for text input questions."""
        if self.client:
            try:
                prompt = self._build_ai_prompt(question, context, response_type="short_text")
                response = await self._call_openai(prompt)
                
                # Clean and limit response length
                cleaned = self._clean_text_response(response)
                return cleaned[:100]  # Limit to 100 characters for text inputs
            except Exception as e:
                logger.warning(f"AI generation failed, using fallback: {e}")
        
        # Fallback: Generate simple response based on question content
        return self._generate_simple_text_response(question.question_text)
    
    async def _generate_long_text_response(self, question: SurveyQuestion, context: Dict[str, Any]) -> str:
        """Generate response for textarea questions."""
        if self.client:
            try:
                prompt = self._build_ai_prompt(question, context, response_type="long_text")
                response = await self._call_openai(prompt)
                
                # Clean and limit response length
                cleaned = self._clean_text_response(response)
                return cleaned[:500]  # Limit to 500 characters for textareas
            except Exception as e:
                logger.warning(f"AI generation failed, using fallback: {e}")
        
        # Fallback: Generate longer response
        return self._generate_detailed_text_response(question.question_text)
    
    async def _generate_rating_response(self, question: SurveyQuestion, context: Dict[str, Any]) -> str:
        """Generate response for rating scale questions."""
        if self.client:
            try:
                prompt = self._build_ai_prompt(question, context, response_type="rating")
                response = await self._call_openai(prompt)
                
                # Extract rating from AI response
                rating = self._extract_rating_from_response(response, question.scale_min, question.scale_max)
                if rating:
                    return str(rating)
            except Exception as e:
                logger.warning(f"AI generation failed, using fallback: {e}")
        
        # Fallback: Generate weighted random rating (slightly positive bias)
        return str(self._generate_weighted_rating(question.scale_min, question.scale_max))
    
    async def _generate_likert_response(self, question: SurveyQuestion, context: Dict[str, Any]) -> str:
        """Generate response for Likert scale questions."""
        # Similar to rating but with specific Likert options
        if question.options:
            return await self._generate_single_choice_response(question, context)
        else:
            return await self._generate_rating_response(question, context)
    
    async def _generate_ranking_response(self, question: SurveyQuestion, context: Dict[str, Any]) -> str:
        """Generate response for ranking questions."""
        if not question.options:
            return ""
        
        if self.client:
            try:
                prompt = self._build_ai_prompt(question, context, response_type="ranking")
                response = await self._call_openai(prompt)
                
                # Extract ranking from AI response
                ranking = self._extract_ranking_from_response(response, question.options)
                if ranking:
                    return ",".join(ranking)
            except Exception as e:
                logger.warning(f"AI generation failed, using fallback: {e}")
        
        # Fallback: Random shuffle with slight preference weighting
        shuffled = question.options.copy()
        random.shuffle(shuffled)
        return ",".join(shuffled)
    
    async def _generate_checkbox_response(self, question: SurveyQuestion, context: Dict[str, Any]) -> str:
        """Generate response for checkbox questions."""
        # Similar to multiple choice
        return await self._generate_multiple_choice_response(question, context)
    
    async def _generate_dropdown_response(self, question: SurveyQuestion, context: Dict[str, Any]) -> str:
        """Generate response for dropdown questions."""
        # Similar to single choice
        return await self._generate_single_choice_response(question, context)
    
    async def _generate_slider_response(self, question: SurveyQuestion, context: Dict[str, Any]) -> str:
        """Generate response for slider questions."""
        # Similar to rating
        return await self._generate_rating_response(question, context)
    
    async def _generate_date_response(self, question: SurveyQuestion, context: Dict[str, Any]) -> str:
        """Generate response for date input questions."""
        # Generate realistic date based on question context
        import datetime
        
        question_lower = question.question_text.lower()
        
        if "birth" in question_lower or "born" in question_lower:
            # Birth date - generate age between 18-70
            age = random.randint(18, 70)
            birth_year = datetime.datetime.now().year - age
            birth_date = datetime.date(birth_year, random.randint(1, 12), random.randint(1, 28))
            return birth_date.strftime("%Y-%m-%d")
        
        elif "recent" in question_lower or "last" in question_lower:
            # Recent date - within last year
            days_ago = random.randint(1, 365)
            recent_date = datetime.date.today() - datetime.timedelta(days=days_ago)
            return recent_date.strftime("%Y-%m-%d")
        
        else:
            # Default: random date within reasonable range
            days_offset = random.randint(-365, 365)
            random_date = datetime.date.today() + datetime.timedelta(days=days_offset)
            return random_date.strftime("%Y-%m-%d")
    
    async def _generate_number_response(self, question: SurveyQuestion, context: Dict[str, Any]) -> str:
        """Generate response for number input questions."""
        question_lower = question.question_text.lower()
        
        if "age" in question_lower:
            return str(random.randint(18, 70))
        elif "income" in question_lower or "salary" in question_lower:
            return str(random.randint(25000, 150000))
        elif "years" in question_lower or "experience" in question_lower:
            return str(random.randint(0, 30))
        elif "hours" in question_lower:
            return str(random.randint(1, 60))
        elif "rating" in question_lower or "score" in question_lower:
            return str(random.randint(1, 10))
        else:
            return str(random.randint(1, 100))
    
    async def _generate_fallback_response(self, question: SurveyQuestion) -> str:
        """Generate a fallback response when specific generators fail."""
        if question.options:
            return random.choice(question.options)
        else:
            return "No response"
    
    def _build_ai_prompt(self, question: SurveyQuestion, context: Dict[str, Any], response_type: str = "general") -> str:
        """Build a prompt for AI response generation."""
        persona = self.current_persona
        
        base_prompt = f"""
You are responding to a survey as a {persona['age_range']} year old person with the following characteristics:
- {persona['characteristics']}
- Preferences: {persona['preferences']}

Question: {question.question_text}
"""
        
        if question.options:
            base_prompt += f"\nAvailable options: {', '.join(question.options)}"
        
        if response_type == "short_text":
            base_prompt += "\nProvide a brief, realistic response (1-10 words)."
        elif response_type == "long_text":
            base_prompt += "\nProvide a detailed but realistic response (2-3 sentences)."
        elif response_type == "rating":
            base_prompt += f"\nProvide a rating between {question.scale_min} and {question.scale_max}."
        elif response_type == "ranking":
            base_prompt += "\nRank the options in order of preference."
        
        base_prompt += "\nRespond naturally as this person would, considering their background and preferences."
        
        return base_prompt
    
    async def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API to generate response."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are helping to complete a survey. Provide realistic, human-like responses."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise
    
    def _extract_options_from_ai_response(self, response: str, options: List[str]) -> List[str]:
        """Extract selected options from AI response."""
        selected = []
        response_lower = response.lower()
        
        for option in options:
            option_lower = option.lower()
            # Check if option is mentioned in response
            if option_lower in response_lower or any(word in response_lower for word in option_lower.split()):
                selected.append(option)
        
        # If no options found, try to match by keywords
        if not selected:
            for option in options:
                option_words = option.lower().split()
                if any(word in response_lower for word in option_words):
                    selected.append(option)
                    break
        
        return selected[:3]  # Limit to 3 selections
    
    def _extract_rating_from_response(self, response: str, min_val: int, max_val: int) -> Optional[int]:
        """Extract rating number from AI response."""
        # Look for numbers in the response
        numbers = re.findall(r'\b(\d+)\b', response)
        
        for num_str in numbers:
            num = int(num_str)
            if min_val <= num <= max_val:
                return num
        
        return None
    
    def _extract_ranking_from_response(self, response: str, options: List[str]) -> List[str]:
        """Extract ranking order from AI response."""
        # Try to find options mentioned in order
        mentioned_options = []
        response_lower = response.lower()
        
        for option in options:
            if option.lower() in response_lower:
                mentioned_options.append(option)
        
        # Add remaining options randomly
        remaining = [opt for opt in options if opt not in mentioned_options]
        random.shuffle(remaining)
        
        return mentioned_options + remaining
    
    def _weighted_random_choice(self, options: List[str]) -> str:
        """Select option with weighted preference for positive/neutral choices."""
        if not options:
            return ""
        
        # Give higher weight to positive/neutral sounding options
        positive_keywords = ['yes', 'good', 'excellent', 'satisfied', 'agree', 'likely', 'often', 'always']
        negative_keywords = ['no', 'bad', 'poor', 'dissatisfied', 'disagree', 'unlikely', 'never', 'rarely']
        
        weights = []
        for option in options:
            option_lower = option.lower()
            weight = 1.0  # Default weight
            
            # Increase weight for positive options
            if any(keyword in option_lower for keyword in positive_keywords):
                weight = 2.0
            # Decrease weight for negative options
            elif any(keyword in option_lower for keyword in negative_keywords):
                weight = 0.5
            
            weights.append(weight)
        
        # Weighted random selection
        return random.choices(options, weights=weights)[0]
    
    def _generate_weighted_rating(self, min_val: int, max_val: int) -> int:
        """Generate rating with slight positive bias."""
        # Create weights favoring higher ratings
        range_size = max_val - min_val + 1
        weights = [i + 1 for i in range(range_size)]  # Linear increase
        
        values = list(range(min_val, max_val + 1))
        return random.choices(values, weights=weights)[0]
    
    def _clean_text_response(self, response: str) -> str:
        """Clean and format text response."""
        # Remove quotes and extra whitespace
        cleaned = response.strip().strip('"\'')
        
        # Remove common AI response prefixes
        prefixes_to_remove = [
            "As a ", "I would say ", "My response is ", "I think ", "In my opinion "
        ]
        
        for prefix in prefixes_to_remove:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):]
        
        return cleaned.strip()
    
    def _generate_simple_text_response(self, question_text: str) -> str:
        """Generate simple text response based on question content."""
        question_lower = question_text.lower()
        
        if "name" in question_lower:
            names = ["John", "Sarah", "Mike", "Lisa", "David", "Emma", "Chris", "Anna"]
            return random.choice(names)
        elif "email" in question_lower:
            domains = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com"]
            return f"user{random.randint(100, 999)}@{random.choice(domains)}"
        elif "phone" in question_lower:
            return f"555-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
        elif "city" in question_lower:
            cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia"]
            return random.choice(cities)
        elif "state" in question_lower:
            states = ["CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA"]
            return random.choice(states)
        else:
            return "Not specified"
    
    def _generate_detailed_text_response(self, question_text: str) -> str:
        """Generate detailed text response for textarea questions."""
        question_lower = question_text.lower()
        
        if "experience" in question_lower or "describe" in question_lower:
            return "I have had positive experiences overall. The service was good and met my expectations. I would recommend it to others."
        elif "improve" in question_lower or "suggestion" in question_lower:
            return "The main areas for improvement would be faster response times and better communication. Overall satisfaction could be enhanced with these changes."
        elif "opinion" in question_lower or "think" in question_lower:
            return "I think this is a good initiative. It addresses important needs and provides value. There's always room for improvement, but the direction is positive."
        else:
            return "This is a thoughtful question. I believe there are multiple perspectives to consider, and the best approach depends on individual circumstances and preferences."
    
    def set_persona(self, persona_name: str) -> bool:
        """Set the current persona for response generation."""
        for persona in self.personas:
            if persona["name"] == persona_name:
                self.current_persona = persona
                logger.info(f"Set persona to: {persona_name}")
                return True
        
        logger.warning(f"Persona not found: {persona_name}")
        return False
    
    def get_available_personas(self) -> List[str]:
        """Get list of available persona names."""
        return [persona["name"] for persona in self.personas]
    
    def add_custom_persona(self, name: str, age_range: str, characteristics: str, preferences: str) -> None:
        """Add a custom persona for response generation."""
        custom_persona = {
            "name": name,
            "age_range": age_range,
            "characteristics": characteristics,
            "preferences": preferences
        }
        
        self.personas.append(custom_persona)
        logger.info(f"Added custom persona: {name}")
    
    async def generate_batch_responses(self, questions: List[SurveyQuestion], context: Dict[str, Any] = None) -> Dict[str, str]:
        """Generate responses for multiple questions at once."""
        responses = {}
        
        for question in questions:
            try:
                response = await self.generate_response(question, context)
                if response:
                    responses[question.question_id] = response
            except Exception as e:
                logger.error(f"Error generating response for question {question.question_id}: {e}")
                continue
        
        logger.info(f"Generated {len(responses)} responses out of {len(questions)} questions")
        return responses

