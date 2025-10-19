"""
Survey Question Extraction Module
Handles extraction and parsing of survey questions from various platforms.
"""

import asyncio
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class QuestionType(Enum):
    """Enumeration of survey question types."""
    MULTIPLE_CHOICE = "multiple_choice"
    SINGLE_CHOICE = "single_choice"
    TEXT_INPUT = "text_input"
    TEXTAREA = "textarea"
    RATING_SCALE = "rating_scale"
    LIKERT_SCALE = "likert_scale"
    RANKING = "ranking"
    CHECKBOX = "checkbox"
    DROPDOWN = "dropdown"
    SLIDER = "slider"
    DATE_INPUT = "date_input"
    NUMBER_INPUT = "number_input"
    UNKNOWN = "unknown"

@dataclass
class SurveyQuestion:
    """Data class representing a survey question."""
    question_id: str
    question_text: str
    question_type: QuestionType
    options: List[str]
    required: bool = False
    min_selections: int = 0
    max_selections: int = 1
    scale_min: int = 1
    scale_max: int = 5
    placeholder: str = ""
    validation_rules: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.validation_rules is None:
            self.validation_rules = {}

class SurveyExtractor:
    """Extracts and parses survey questions from web pages."""
    
    def __init__(self, browser_manager):
        self.browser = browser_manager
        self.current_questions = []
        
    async def extract_questions_from_page(self) -> List[SurveyQuestion]:
        """Extract all questions from the current page."""
        try:
            if not self.browser.page:
                logger.error("No browser page available")
                return []
            
            # Get page content
            content = await self.browser.get_page_content()
            soup = BeautifulSoup(content, 'html.parser')
            
            questions = []
            
            # Try different extraction methods
            questions.extend(await self._extract_form_questions())
            questions.extend(await self._extract_survey_specific_questions())
            questions.extend(await self._extract_generic_questions(soup))
            
            # Remove duplicates based on question_id
            unique_questions = []
            seen_ids = set()
            for question in questions:
                if question.question_id not in seen_ids:
                    unique_questions.append(question)
                    seen_ids.add(question.question_id)
            
            self.current_questions = unique_questions
            logger.info(f"Extracted {len(unique_questions)} unique questions from page")
            
            return unique_questions
            
        except Exception as e:
            logger.error(f"Error extracting questions: {e}")
            return []
    
    async def _extract_form_questions(self) -> List[SurveyQuestion]:
        """Extract questions from standard HTML forms."""
        questions = []
        
        try:
            # Find all form elements
            forms = await self.browser.page.query_selector_all('form')
            
            for form in forms:
                # Look for various input types
                inputs = await form.query_selector_all('input, select, textarea')
                
                for input_elem in inputs:
                    try:
                        question = await self._parse_form_element(input_elem)
                        if question:
                            questions.append(question)
                    except Exception as e:
                        logger.warning(f"Error parsing form element: {e}")
                        continue
            
        except Exception as e:
            logger.error(f"Error extracting form questions: {e}")
        
        return questions
    
    async def _parse_form_element(self, element) -> Optional[SurveyQuestion]:
        """Parse a single form element into a SurveyQuestion."""
        try:
            # Get element attributes
            tag_name = await element.evaluate('el => el.tagName.toLowerCase()')
            input_type = await element.get_attribute('type') or ''
            name = await element.get_attribute('name') or ''
            id_attr = await element.get_attribute('id') or ''
            required = await element.get_attribute('required') is not None
            placeholder = await element.get_attribute('placeholder') or ''
            
            # Skip non-question elements
            skip_types = ['submit', 'button', 'hidden', 'csrf_token']
            if input_type.lower() in skip_types or not (name or id_attr):
                return None
            
            # Generate question ID
            question_id = id_attr or name or f"question_{hash(str(element))}"
            
            # Find associated label or question text
            question_text = await self._find_question_text(element)
            
            # Determine question type and extract options
            question_type, options = await self._determine_question_type(element, tag_name, input_type)
            
            # Extract additional properties
            min_val = await element.get_attribute('min')
            max_val = await element.get_attribute('max')
            
            question = SurveyQuestion(
                question_id=question_id,
                question_text=question_text,
                question_type=question_type,
                options=options,
                required=required,
                placeholder=placeholder
            )
            
            # Set scale values for rating questions
            if question_type in [QuestionType.RATING_SCALE, QuestionType.SLIDER]:
                if min_val and min_val.isdigit():
                    question.scale_min = int(min_val)
                if max_val and max_val.isdigit():
                    question.scale_max = int(max_val)
            
            return question
            
        except Exception as e:
            logger.error(f"Error parsing form element: {e}")
            return None
    
    async def _find_question_text(self, element) -> str:
        """Find the question text associated with a form element."""
        try:
            # Try to find associated label
            id_attr = await element.get_attribute('id')
            if id_attr:
                label = await self.browser.page.query_selector(f'label[for="{id_attr}"]')
                if label:
                    text = await label.inner_text()
                    if text.strip():
                        return text.strip()
            
            # Look for nearby text elements
            parent = await element.evaluate('el => el.parentElement')
            if parent:
                # Check for question text in parent or siblings
                text_elements = await parent.query_selector_all('span, div, p, h1, h2, h3, h4, h5, h6')
                for text_elem in text_elements:
                    text = await text_elem.inner_text()
                    if text and len(text.strip()) > 5 and '?' in text:
                        return text.strip()
            
            # Fallback to placeholder or name
            placeholder = await element.get_attribute('placeholder')
            if placeholder:
                return placeholder
            
            name = await element.get_attribute('name')
            if name:
                return name.replace('_', ' ').title()
            
            return "Unknown Question"
            
        except Exception as e:
            logger.error(f"Error finding question text: {e}")
            return "Unknown Question"
    
    async def _determine_question_type(self, element, tag_name: str, input_type: str) -> Tuple[QuestionType, List[str]]:
        """Determine the question type and extract options if applicable."""
        try:
            options = []
            
            if tag_name == 'select':
                # Dropdown/select element
                option_elements = await element.query_selector_all('option')
                for option in option_elements:
                    text = await option.inner_text()
                    value = await option.get_attribute('value')
                    if text.strip() and value:
                        options.append(text.strip())
                
                return QuestionType.DROPDOWN, options
            
            elif tag_name == 'textarea':
                return QuestionType.TEXTAREA, []
            
            elif tag_name == 'input':
                if input_type in ['radio']:
                    # Radio button - part of single choice
                    name = await element.get_attribute('name')
                    if name:
                        # Find all radio buttons with same name
                        radios = await self.browser.page.query_selector_all(f'input[name="{name}"][type="radio"]')
                        for radio in radios:
                            label_text = await self._find_question_text(radio)
                            if label_text and label_text not in options:
                                options.append(label_text)
                    
                    return QuestionType.SINGLE_CHOICE, options
                
                elif input_type in ['checkbox']:
                    # Checkbox - could be multiple choice
                    name = await element.get_attribute('name')
                    if name:
                        # Find all checkboxes with same name
                        checkboxes = await self.browser.page.query_selector_all(f'input[name="{name}"][type="checkbox"]')
                        for checkbox in checkboxes:
                            label_text = await self._find_question_text(checkbox)
                            if label_text and label_text not in options:
                                options.append(label_text)
                    
                    return QuestionType.CHECKBOX, options
                
                elif input_type in ['range']:
                    return QuestionType.SLIDER, []
                
                elif input_type in ['number']:
                    return QuestionType.NUMBER_INPUT, []
                
                elif input_type in ['date', 'datetime-local']:
                    return QuestionType.DATE_INPUT, []
                
                elif input_type in ['text', 'email', 'tel', 'url']:
                    return QuestionType.TEXT_INPUT, []
                
                else:
                    return QuestionType.UNKNOWN, []
            
            return QuestionType.UNKNOWN, []
            
        except Exception as e:
            logger.error(f"Error determining question type: {e}")
            return QuestionType.UNKNOWN, []
    
    async def _extract_survey_specific_questions(self) -> List[SurveyQuestion]:
        """Extract questions using survey-specific selectors."""
        questions = []
        
        try:
            # Common survey question selectors
            question_selectors = [
                '.question',
                '.survey-question',
                '[data-question]',
                '.question-container',
                '.form-question',
                '.quiz-question'
            ]
            
            for selector in question_selectors:
                question_elements = await self.browser.page.query_selector_all(selector)
                
                for elem in question_elements:
                    try:
                        question = await self._parse_survey_question_element(elem)
                        if question:
                            questions.append(question)
                    except Exception as e:
                        logger.warning(f"Error parsing survey question element: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Error extracting survey-specific questions: {e}")
        
        return questions
    
    async def _parse_survey_question_element(self, element) -> Optional[SurveyQuestion]:
        """Parse a survey-specific question element."""
        try:
            # Extract question text
            text_selectors = ['.question-text', '.question-title', 'h1', 'h2', 'h3', 'h4', 'p']
            question_text = "Unknown Question"
            
            for selector in text_selectors:
                text_elem = await element.query_selector(selector)
                if text_elem:
                    text = await text_elem.inner_text()
                    if text.strip():
                        question_text = text.strip()
                        break
            
            # Generate question ID
            id_attr = await element.get_attribute('id')
            data_id = await element.get_attribute('data-question-id')
            question_id = id_attr or data_id or f"survey_q_{hash(question_text)}"
            
            # Look for answer options
            options = []
            option_selectors = [
                '.option', '.answer', '.choice',
                'input[type="radio"]', 'input[type="checkbox"]',
                'select option'
            ]
            
            for selector in option_selectors:
                option_elements = await element.query_selector_all(selector)
                for option in option_elements:
                    try:
                        if 'input' in selector:
                            # For input elements, get associated label
                            option_text = await self._find_question_text(option)
                        else:
                            option_text = await option.inner_text()
                        
                        if option_text.strip() and option_text not in options:
                            options.append(option_text.strip())
                    except:
                        continue
            
            # Determine question type based on found elements
            question_type = QuestionType.UNKNOWN
            
            if await element.query_selector('input[type="radio"]'):
                question_type = QuestionType.SINGLE_CHOICE
            elif await element.query_selector('input[type="checkbox"]'):
                question_type = QuestionType.CHECKBOX
            elif await element.query_selector('select'):
                question_type = QuestionType.DROPDOWN
            elif await element.query_selector('textarea'):
                question_type = QuestionType.TEXTAREA
            elif await element.query_selector('input[type="text"]'):
                question_type = QuestionType.TEXT_INPUT
            elif await element.query_selector('input[type="range"]'):
                question_type = QuestionType.SLIDER
            elif len(options) > 1:
                question_type = QuestionType.MULTIPLE_CHOICE
            
            # Check if required
            required = await element.get_attribute('required') is not None
            required_elem = await element.query_selector('.required, [required]')
            if required_elem:
                required = True
            
            return SurveyQuestion(
                question_id=question_id,
                question_text=question_text,
                question_type=question_type,
                options=options,
                required=required
            )
            
        except Exception as e:
            logger.error(f"Error parsing survey question element: {e}")
            return None
    
    async def _extract_generic_questions(self, soup: BeautifulSoup) -> List[SurveyQuestion]:
        """Extract questions using generic text analysis."""
        questions = []
        
        try:
            # Find text that looks like questions
            text_elements = soup.find_all(['p', 'div', 'span', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            
            for elem in text_elements:
                text = elem.get_text().strip()
                
                # Check if text looks like a question
                if self._is_question_text(text):
                    question_id = f"generic_q_{hash(text)}"
                    
                    # Look for nearby input elements
                    options = []
                    question_type = QuestionType.TEXT_INPUT  # Default
                    
                    # Find parent container and look for inputs
                    parent = elem.parent
                    if parent:
                        inputs = parent.find_all(['input', 'select', 'textarea'])
                        if inputs:
                            # Determine type based on first input found
                            first_input = inputs[0]
                            input_type = first_input.get('type', '')
                            
                            if first_input.name == 'select':
                                question_type = QuestionType.DROPDOWN
                                options = [opt.get_text().strip() for opt in first_input.find_all('option') if opt.get_text().strip()]
                            elif input_type == 'radio':
                                question_type = QuestionType.SINGLE_CHOICE
                            elif input_type == 'checkbox':
                                question_type = QuestionType.CHECKBOX
                            elif first_input.name == 'textarea':
                                question_type = QuestionType.TEXTAREA
                    
                    question = SurveyQuestion(
                        question_id=question_id,
                        question_text=text,
                        question_type=question_type,
                        options=options
                    )
                    
                    questions.append(question)
        
        except Exception as e:
            logger.error(f"Error extracting generic questions: {e}")
        
        return questions
    
    def _is_question_text(self, text: str) -> bool:
        """Determine if text looks like a survey question."""
        if not text or len(text) < 10:
            return False
        
        # Question indicators
        question_indicators = [
            '?', 'how', 'what', 'when', 'where', 'why', 'which', 'who',
            'do you', 'are you', 'have you', 'would you', 'could you',
            'please', 'rate', 'select', 'choose', 'indicate'
        ]
        
        text_lower = text.lower()
        
        # Check for question marks or question words
        if '?' in text:
            return True
        
        # Check for question indicators
        for indicator in question_indicators:
            if indicator in text_lower:
                return True
        
        return False
    
    async def get_current_question(self) -> Optional[SurveyQuestion]:
        """Get the current active question on the page."""
        try:
            # Look for focused or highlighted question
            active_selectors = [
                '.question.active',
                '.question.current',
                '.question:focus-within',
                '[data-current="true"]'
            ]
            
            for selector in active_selectors:
                element = await self.browser.page.query_selector(selector)
                if element:
                    question = await self._parse_survey_question_element(element)
                    if question:
                        return question
            
            # If no active question found, return first question
            if self.current_questions:
                return self.current_questions[0]
            
            # Extract questions if none cached
            questions = await self.extract_questions_from_page()
            return questions[0] if questions else None
            
        except Exception as e:
            logger.error(f"Error getting current question: {e}")
            return None
    
    async def is_survey_complete(self) -> bool:
        """Check if the survey is complete."""
        try:
            # Look for completion indicators
            completion_indicators = [
                'thank you',
                'survey complete',
                'completed',
                'finished',
                'thank you for participating',
                'your responses have been recorded'
            ]
            
            page_text = await self.browser.page.inner_text('body')
            page_text_lower = page_text.lower()
            
            for indicator in completion_indicators:
                if indicator in page_text_lower:
                    return True
            
            # Check for completion URLs
            current_url = self.browser.page.url.lower()
            completion_url_indicators = ['complete', 'finish', 'thank', 'done']
            
            for indicator in completion_url_indicators:
                if indicator in current_url:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking survey completion: {e}")
            return False
    
    def get_question_summary(self) -> Dict[str, Any]:
        """Get a summary of extracted questions."""
        if not self.current_questions:
            return {"total": 0, "by_type": {}}
        
        type_counts = {}
        for question in self.current_questions:
            question_type = question.question_type.value
            type_counts[question_type] = type_counts.get(question_type, 0) + 1
        
        return {
            "total": len(self.current_questions),
            "by_type": type_counts,
            "questions": [
                {
                    "id": q.question_id,
                    "text": q.question_text[:100] + "..." if len(q.question_text) > 100 else q.question_text,
                    "type": q.question_type.value,
                    "options_count": len(q.options),
                    "required": q.required
                }
                for q in self.current_questions
            ]
        }

