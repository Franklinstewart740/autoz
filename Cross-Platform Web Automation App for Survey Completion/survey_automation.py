"""
Survey Automation Orchestrator
Main class that coordinates all automation components for survey completion.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import traceback

from .browser_manager import BrowserManager
from .proxy_manager import ProxyManager
from .captcha_solver import CaptchaSolver
from .platform_handlers import get_platform_handler, BasePlatformHandler
from .survey_extractor import SurveyExtractor, SurveyQuestion, QuestionType
from .ai_response_generator import AIResponseGenerator

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    """Enumeration of task statuses."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"

@dataclass
class SurveyTask:
    """Data class representing a survey automation task."""
    task_id: str
    platform: str
    credentials: Dict[str, str]
    survey_id: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    current_step: str = ""
    error_message: str = ""
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    responses_submitted: int = 0
    total_questions: int = 0
    retry_count: int = 0
    max_retries: int = 3

class SurveyAutomation:
    """Main survey automation orchestrator."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.tasks = {}
        self.running_tasks = set()
        
        # Initialize components
        self.proxy_manager = ProxyManager()
        self.captcha_solver = CaptchaSolver(
            api_key=self.config.get("captcha_api_key"),
            service=self.config.get("captcha_service", "2captcha")
        )
        self.ai_generator = AIResponseGenerator(
            api_key=self.config.get("openai_api_key"),
            model=self.config.get("ai_model", "gpt-3.5-turbo")
        )
        
        # Error handling configuration
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_delay = self.config.get("retry_delay", 5)
        self.timeout_seconds = self.config.get("timeout_seconds", 300)
        
        logger.info("Survey automation system initialized")
    
    async def create_task(self, platform: str, credentials: Dict[str, str], survey_id: Optional[str] = None) -> str:
        """Create a new survey automation task.
        
        Args:
            platform: Platform name ("swagbucks" or "inboxdollars")
            credentials: Login credentials
            survey_id: Optional specific survey ID to complete
            
        Returns:
            Task ID
        """
        task_id = f"{platform}_{int(time.time())}_{len(self.tasks)}"
        
        task = SurveyTask(
            task_id=task_id,
            platform=platform.lower(),
            credentials=credentials,
            survey_id=survey_id,
            max_retries=self.max_retries
        )
        
        self.tasks[task_id] = task
        logger.info(f"Created task {task_id} for platform {platform}")
        
        return task_id
    
    async def start_task(self, task_id: str) -> bool:
        """Start executing a survey automation task.
        
        Args:
            task_id: ID of the task to start
            
        Returns:
            True if task started successfully, False otherwise
        """
        if task_id not in self.tasks:
            logger.error(f"Task {task_id} not found")
            return False
        
        task = self.tasks[task_id]
        
        if task.status != TaskStatus.PENDING:
            logger.error(f"Task {task_id} is not in pending status")
            return False
        
        if task_id in self.running_tasks:
            logger.error(f"Task {task_id} is already running")
            return False
        
        # Start task execution in background
        asyncio.create_task(self._execute_task(task))
        
        return True
    
    async def _execute_task(self, task: SurveyTask) -> None:
        """Execute a survey automation task with error handling and retries."""
        task.status = TaskStatus.RUNNING
        task.start_time = time.time()
        task.current_step = "Initializing"
        self.running_tasks.add(task.task_id)
        
        browser_manager = None
        platform_handler = None
        
        try:
            logger.info(f"Starting execution of task {task.task_id}")
            
            # Initialize browser with proxy if configured
            proxy_config = None
            if self.config.get("use_proxy", False):
                proxy_config = self.proxy_manager.get_current_proxy()
                if proxy_config:
                    proxy_config = self.proxy_manager.configure_for_playwright(proxy_config)
            
            browser_manager = BrowserManager(proxy_config)
            await browser_manager.start_browser(
                headless=self.config.get("headless", True),
                browser_type=self.config.get("browser_type", "chromium")
            )
            
            # Initialize platform handler
            platform_handler = get_platform_handler(
                task.platform, browser_manager, self.captcha_solver
            )
            
            if not platform_handler:
                raise Exception(f"Unsupported platform: {task.platform}")
            
            # Execute survey automation steps
            await self._execute_survey_steps(task, platform_handler, browser_manager)
            
            # Mark task as completed
            task.status = TaskStatus.COMPLETED
            task.current_step = "Completed"
            task.progress = 100.0
            task.end_time = time.time()
            
            logger.info(f"Task {task.task_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")
            logger.error(traceback.format_exc())
            
            task.error_message = str(e)
            task.retry_count += 1
            
            # Retry logic
            if task.retry_count <= task.max_retries:
                logger.info(f"Retrying task {task.task_id} (attempt {task.retry_count}/{task.max_retries})")
                task.status = TaskStatus.PENDING
                task.current_step = "Retrying"
                
                # Wait before retry
                await asyncio.sleep(self.retry_delay * task.retry_count)
                
                # Retry the task
                asyncio.create_task(self._execute_task(task))
            else:
                task.status = TaskStatus.FAILED
                task.current_step = "Failed"
                task.end_time = time.time()
                logger.error(f"Task {task.task_id} failed after {task.max_retries} retries")
        
        finally:
            # Cleanup resources
            if browser_manager:
                try:
                    await browser_manager.close()
                except Exception as e:
                    logger.error(f"Error closing browser: {e}")
            
            self.running_tasks.discard(task.task_id)
    
    async def _execute_survey_steps(self, task: SurveyTask, platform_handler: BasePlatformHandler, browser_manager: BrowserManager) -> None:
        """Execute the main survey automation steps."""
        
        # Step 1: Login
        task.current_step = "Logging in"
        task.progress = 10.0
        
        login_success = await platform_handler.login(task.credentials)
        if not login_success:
            raise Exception("Login failed")
        
        logger.info(f"Successfully logged into {task.platform}")
        
        # Step 2: Navigate to surveys
        task.current_step = "Navigating to surveys"
        task.progress = 20.0
        
        nav_success = await platform_handler.navigate_to_surveys()
        if not nav_success:
            raise Exception("Failed to navigate to surveys")
        
        # Step 3: Get available surveys or start specific survey
        task.current_step = "Finding surveys"
        task.progress = 30.0
        
        if task.survey_id:
            # Start specific survey
            survey_started = await platform_handler.start_survey(task.survey_id)
            if not survey_started:
                raise Exception(f"Failed to start survey {task.survey_id}")
        else:
            # Get available surveys and start the first one
            surveys = await platform_handler.get_available_surveys()
            if not surveys:
                raise Exception("No surveys available")
            
            # Start the first available survey
            first_survey = surveys[0]
            survey_url = first_survey.get("url")
            if survey_url:
                survey_started = await platform_handler.start_survey(survey_url)
                if not survey_started:
                    raise Exception("Failed to start survey")
        
        # Step 4: Complete survey
        task.current_step = "Completing survey"
        task.progress = 40.0
        
        await self._complete_survey(task, browser_manager)
        
        logger.info(f"Survey completion finished for task {task.task_id}")
    
    async def _complete_survey(self, task: SurveyTask, browser_manager: BrowserManager) -> None:
        """Complete the survey by answering questions."""
        survey_extractor = SurveyExtractor(browser_manager)
        max_questions = 50  # Prevent infinite loops
        questions_answered = 0
        
        while questions_answered < max_questions:
            try:
                # Check if survey is complete
                if await survey_extractor.is_survey_complete():
                    logger.info("Survey completed successfully")
                    break
                
                # Extract current question
                current_question = await survey_extractor.get_current_question()
                if not current_question:
                    # Try to extract all questions from page
                    questions = await survey_extractor.extract_questions_from_page()
                    if not questions:
                        logger.warning("No questions found on page")
                        break
                    current_question = questions[0]
                
                logger.info(f"Processing question: {current_question.question_text[:100]}...")
                
                # Generate response using AI
                response = await self.ai_generator.generate_response(current_question)
                if not response:
                    logger.warning("Failed to generate response, skipping question")
                    continue
                
                # Submit response
                success = await self._submit_response(browser_manager, current_question, response)
                if success:
                    questions_answered += 1
                    task.responses_submitted = questions_answered
                    task.progress = min(40.0 + (questions_answered / max_questions) * 50.0, 90.0)
                    
                    logger.info(f"Successfully submitted response {questions_answered}")
                else:
                    logger.warning("Failed to submit response")
                
                # Wait between questions to appear human-like
                await asyncio.sleep(random.uniform(2, 5))
                
                # Handle any CAPTCHAs that might appear
                captcha_type = await self.captcha_solver.detect_captcha_type(browser_manager.page)
                if captcha_type:
                    logger.info(f"CAPTCHA detected: {captcha_type}")
                    solution = await self.captcha_solver.solve_captcha(browser_manager.page, captcha_type)
                    if solution:
                        await self.captcha_solver.submit_captcha_solution(
                            browser_manager.page, captcha_type, solution
                        )
                
            except Exception as e:
                logger.error(f"Error processing question: {e}")
                # Continue to next question
                continue
        
        task.total_questions = questions_answered
        logger.info(f"Completed survey with {questions_answered} questions answered")
    
    async def _submit_response(self, browser_manager: BrowserManager, question: SurveyQuestion, response: str) -> bool:
        """Submit a response to a survey question.
        
        Args:
            browser_manager: Browser manager instance
            question: The survey question
            response: The generated response
            
        Returns:
            True if submission successful, False otherwise
        """
        try:
            page = browser_manager.page
            
            if question.question_type == QuestionType.SINGLE_CHOICE:
                # Handle radio buttons
                if question.options and response in question.options:
                    # Find radio button for the selected option
                    radio_selector = f'input[type="radio"][value="{response}"], input[type="radio"] + label:has-text("{response}")'
                    if await browser_manager.wait_for_element(radio_selector, timeout=5000):
                        await browser_manager.human_like_click(radio_selector)
                        return True
            
            elif question.question_type == QuestionType.MULTIPLE_CHOICE or question.question_type == QuestionType.CHECKBOX:
                # Handle checkboxes
                selected_options = response.split(",")
                for option in selected_options:
                    option = option.strip()
                    if option in question.options:
                        checkbox_selector = f'input[type="checkbox"][value="{option}"], input[type="checkbox"] + label:has-text("{option}")'
                        if await browser_manager.wait_for_element(checkbox_selector, timeout=2000):
                            await browser_manager.human_like_click(checkbox_selector)
                
                return True
            
            elif question.question_type == QuestionType.DROPDOWN:
                # Handle select dropdowns
                select_selector = f'select[name="{question.question_id}"], select[id="{question.question_id}"]'
                if await browser_manager.wait_for_element(select_selector, timeout=5000):
                    await page.select_option(select_selector, response)
                    return True
            
            elif question.question_type in [QuestionType.TEXT_INPUT, QuestionType.TEXTAREA]:
                # Handle text inputs
                input_selector = f'input[name="{question.question_id}"], input[id="{question.question_id}"], textarea[name="{question.question_id}"], textarea[id="{question.question_id}"]'
                if await browser_manager.wait_for_element(input_selector, timeout=5000):
                    await browser_manager.human_like_type(input_selector, response)
                    return True
            
            elif question.question_type in [QuestionType.RATING_SCALE, QuestionType.SLIDER]:
                # Handle sliders and rating scales
                slider_selector = f'input[type="range"][name="{question.question_id}"], input[type="range"][id="{question.question_id}"]'
                if await browser_manager.wait_for_element(slider_selector, timeout=5000):
                    await page.fill(slider_selector, response)
                    return True
            
            elif question.question_type == QuestionType.NUMBER_INPUT:
                # Handle number inputs
                number_selector = f'input[type="number"][name="{question.question_id}"], input[type="number"][id="{question.question_id}"]'
                if await browser_manager.wait_for_element(number_selector, timeout=5000):
                    await browser_manager.human_like_type(number_selector, response)
                    return True
            
            elif question.question_type == QuestionType.DATE_INPUT:
                # Handle date inputs
                date_selector = f'input[type="date"][name="{question.question_id}"], input[type="date"][id="{question.question_id}"]'
                if await browser_manager.wait_for_element(date_selector, timeout=5000):
                    await page.fill(date_selector, response)
                    return True
            
            # Try generic submission methods if specific type handling failed
            return await self._try_generic_submission(browser_manager, question, response)
            
        except Exception as e:
            logger.error(f"Error submitting response: {e}")
            return False
    
    async def _try_generic_submission(self, browser_manager: BrowserManager, question: SurveyQuestion, response: str) -> bool:
        """Try generic methods to submit response when specific type handling fails."""
        try:
            page = browser_manager.page
            
            # Try to find any input associated with the question
            possible_selectors = [
                f'[name*="{question.question_id}"]',
                f'[id*="{question.question_id}"]',
                f'[data-question="{question.question_id}"]',
                'input:not([type="hidden"]):not([type="submit"]):not([type="button"])',
                'select',
                'textarea'
            ]
            
            for selector in possible_selectors:
                elements = await page.query_selector_all(selector)
                for element in elements:
                    try:
                        tag_name = await element.evaluate('el => el.tagName.toLowerCase()')
                        input_type = await element.get_attribute('type') or ''
                        
                        if tag_name == 'input':
                            if input_type in ['text', 'email', 'tel', 'url', 'number']:
                                await element.fill(response)
                                return True
                            elif input_type == 'radio' and response in question.options:
                                # Check if this radio button matches our response
                                value = await element.get_attribute('value')
                                if value == response:
                                    await element.click()
                                    return True
                            elif input_type == 'checkbox':
                                # For checkboxes, check if response contains this option
                                value = await element.get_attribute('value')
                                if value and value in response:
                                    await element.click()
                        
                        elif tag_name == 'select':
                            await element.select_option(response)
                            return True
                        
                        elif tag_name == 'textarea':
                            await element.fill(response)
                            return True
                    
                    except Exception:
                        continue
            
            return False
            
        except Exception as e:
            logger.error(f"Generic submission failed: {e}")
            return False
    
    async def _find_and_click_next_button(self, browser_manager: BrowserManager) -> bool:
        """Find and click the next/continue button."""
        try:
            next_button_selectors = [
                'button:has-text("Next")',
                'button:has-text("Continue")',
                'button:has-text("Submit")',
                'input[type="submit"]',
                'button[type="submit"]',
                '.next-button',
                '.continue-button',
                '.submit-button',
                '[data-action="next"]',
                '[data-action="continue"]'
            ]
            
            for selector in next_button_selectors:
                if await browser_manager.wait_for_element(selector, timeout=2000):
                    await browser_manager.human_like_click(selector)
                    await asyncio.sleep(2)  # Wait for page transition
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error clicking next button: {e}")
            return False
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Task status dictionary or None if task not found
        """
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id]
        
        return {
            "task_id": task.task_id,
            "platform": task.platform,
            "status": task.status.value,
            "progress": task.progress,
            "current_step": task.current_step,
            "error_message": task.error_message,
            "start_time": task.start_time,
            "end_time": task.end_time,
            "responses_submitted": task.responses_submitted,
            "total_questions": task.total_questions,
            "retry_count": task.retry_count,
            "max_retries": task.max_retries
        }
    
    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Get status of all tasks."""
        return [self.get_task_status(task_id) for task_id in self.tasks.keys()]
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task.
        
        Args:
            task_id: ID of the task to cancel
            
        Returns:
            True if task was cancelled, False otherwise
        """
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        
        if task.status == TaskStatus.RUNNING:
            task.status = TaskStatus.CANCELLED
            task.current_step = "Cancelled"
            task.end_time = time.time()
            self.running_tasks.discard(task_id)
            
            logger.info(f"Task {task_id} cancelled")
            return True
        
        return False
    
    def configure_proxy(self, proxy_list: List[str], username: str = None, password: str = None) -> None:
        """Configure proxy settings.
        
        Args:
            proxy_list: List of proxy server addresses
            username: Optional proxy username
            password: Optional proxy password
        """
        self.proxy_manager.add_proxies_from_list(proxy_list, username, password)
        logger.info(f"Configured {len(proxy_list)} proxies")
    
    def set_ai_persona(self, persona_name: str) -> bool:
        """Set the AI persona for response generation.
        
        Args:
            persona_name: Name of the persona to use
            
        Returns:
            True if persona was set, False if not found
        """
        return self.ai_generator.set_persona(persona_name)
    
    def get_available_personas(self) -> List[str]:
        """Get list of available AI personas."""
        return self.ai_generator.get_available_personas()

# Import random for human-like delays
import random

