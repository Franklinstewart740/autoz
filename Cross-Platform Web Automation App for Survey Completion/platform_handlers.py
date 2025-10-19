"""
Platform-specific handlers for survey sites.
Handles login, navigation, and platform-specific logic for Swagbucks and InboxDollars.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from .browser_manager import BrowserManager
from .captcha_solver import CaptchaSolver

logger = logging.getLogger(__name__)

class BasePlatformHandler(ABC):
    """Abstract base class for platform handlers."""
    
    def __init__(self, browser_manager: BrowserManager, captcha_solver: CaptchaSolver):
        self.browser = browser_manager
        self.captcha_solver = captcha_solver
        self.is_logged_in = False
        self.platform_name = ""
        
    @abstractmethod
    async def login(self, credentials: Dict[str, str]) -> bool:
        """Login to the platform."""
        pass
    
    @abstractmethod
    async def navigate_to_surveys(self) -> bool:
        """Navigate to the surveys section."""
        pass
    
    @abstractmethod
    async def get_available_surveys(self) -> List[Dict[str, Any]]:
        """Get list of available surveys."""
        pass
    
    @abstractmethod
    async def start_survey(self, survey_id: str) -> bool:
        """Start a specific survey."""
        pass
    
    async def handle_captcha_if_present(self) -> bool:
        """Check for and handle CAPTCHA if present."""
        try:
            captcha_type = await self.captcha_solver.detect_captcha_type(self.browser.page)
            if captcha_type:
                logger.info(f"CAPTCHA detected: {captcha_type}")
                solution = await self.captcha_solver.solve_captcha(self.browser.page, captcha_type)
                if solution:
                    success = await self.captcha_solver.submit_captcha_solution(
                        self.browser.page, captcha_type, solution
                    )
                    if success:
                        logger.info("CAPTCHA solved successfully")
                        return True
                    else:
                        logger.error("Failed to submit CAPTCHA solution")
                        return False
                else:
                    logger.error("Failed to solve CAPTCHA")
                    return False
            return True  # No CAPTCHA present
        except Exception as e:
            logger.error(f"Error handling CAPTCHA: {e}")
            return False

class SwagbucksHandler(BasePlatformHandler):
    """Handler for Swagbucks platform."""
    
    def __init__(self, browser_manager: BrowserManager, captcha_solver: CaptchaSolver):
        super().__init__(browser_manager, captcha_solver)
        self.platform_name = "Swagbucks"
        self.base_url = "https://www.swagbucks.com"
        self.login_url = "https://www.swagbucks.com/p/login"
        self.surveys_url = "https://www.swagbucks.com/surveys"
        
    async def login(self, credentials: Dict[str, str]) -> bool:
        """Login to Swagbucks."""
        try:
            email = credentials.get("email")
            password = credentials.get("password")
            
            if not email or not password:
                logger.error("Email and password required for Swagbucks login")
                return False
            
            # Navigate to login page
            await self.browser.navigate_to(self.login_url)
            
            # Handle CAPTCHA if present
            if not await self.handle_captcha_if_present():
                return False
            
            # Fill login form
            email_selector = 'input[name="emailAddress"], input[id="emailAddress"], input[type="email"]'
            password_selector = 'input[name="password"], input[id="password"], input[type="password"]'
            
            # Wait for form elements
            if not await self.browser.wait_for_element(email_selector, timeout=10000):
                logger.error("Could not find email input field")
                return False
            
            # Fill credentials
            await self.browser.human_like_type(email_selector, email)
            await asyncio.sleep(1)
            await self.browser.human_like_type(password_selector, password)
            
            # Handle CAPTCHA again if it appeared after typing
            if not await self.handle_captcha_if_present():
                return False
            
            # Submit form
            submit_selectors = [
                'button[type="submit"]',
                'input[type="submit"]',
                'button:contains("Sign In")',
                'button:contains("Login")',
                '.login-button',
                '#loginButton'
            ]
            
            submitted = False
            for selector in submit_selectors:
                try:
                    if await self.browser.wait_for_element(selector, timeout=2000):
                        await self.browser.human_like_click(selector)
                        submitted = True
                        break
                except:
                    continue
            
            if not submitted:
                # Try pressing Enter on password field
                await self.browser.page.press(password_selector, "Enter")
            
            # Wait for login to complete
            await asyncio.sleep(3)
            
            # Check if login was successful
            current_url = self.browser.page.url
            if "login" not in current_url.lower() and "swagbucks.com" in current_url:
                # Look for user-specific elements
                user_indicators = [
                    '[data-testid="user-menu"]',
                    '.user-name',
                    '.account-menu',
                    '.member-dashboard'
                ]
                
                for indicator in user_indicators:
                    if await self.browser.wait_for_element(indicator, timeout=5000):
                        self.is_logged_in = True
                        logger.info("Successfully logged into Swagbucks")
                        return True
            
            logger.error("Swagbucks login failed - still on login page or no user indicators found")
            return False
            
        except Exception as e:
            logger.error(f"Error during Swagbucks login: {e}")
            return False
    
    async def navigate_to_surveys(self) -> bool:
        """Navigate to Swagbucks surveys section."""
        try:
            if not self.is_logged_in:
                logger.error("Must be logged in to navigate to surveys")
                return False
            
            # Navigate to surveys page
            await self.browser.navigate_to(self.surveys_url)
            
            # Wait for surveys to load
            survey_indicators = [
                '.survey-item',
                '.survey-card',
                '[data-testid="survey"]',
                '.available-surveys'
            ]
            
            for indicator in survey_indicators:
                if await self.browser.wait_for_element(indicator, timeout=10000):
                    logger.info("Successfully navigated to Swagbucks surveys")
                    return True
            
            logger.warning("Navigated to surveys page but no survey elements found")
            return True  # Page loaded, even if no surveys visible
            
        except Exception as e:
            logger.error(f"Error navigating to Swagbucks surveys: {e}")
            return False
    
    async def get_available_surveys(self) -> List[Dict[str, Any]]:
        """Get list of available surveys from Swagbucks."""
        try:
            surveys = []
            
            # Look for survey elements
            survey_selectors = [
                '.survey-item',
                '.survey-card',
                '[data-testid="survey"]',
                '.offer-item'
            ]
            
            for selector in survey_selectors:
                survey_elements = await self.browser.page.query_selector_all(selector)
                if survey_elements:
                    for element in survey_elements:
                        try:
                            # Extract survey information
                            title_elem = await element.query_selector('.title, .survey-title, h3, h4')
                            reward_elem = await element.query_selector('.reward, .points, .sb-amount')
                            time_elem = await element.query_selector('.time, .duration, .minutes')
                            
                            title = await title_elem.inner_text() if title_elem else "Unknown Survey"
                            reward = await reward_elem.inner_text() if reward_elem else "Unknown Reward"
                            time_est = await time_elem.inner_text() if time_elem else "Unknown Time"
                            
                            # Get survey link or ID
                            link_elem = await element.query_selector('a')
                            survey_url = await link_elem.get_attribute('href') if link_elem else None
                            
                            survey_data = {
                                "title": title.strip(),
                                "reward": reward.strip(),
                                "estimated_time": time_est.strip(),
                                "url": survey_url,
                                "platform": "Swagbucks"
                            }
                            
                            surveys.append(survey_data)
                            
                        except Exception as e:
                            logger.warning(f"Error extracting survey data: {e}")
                            continue
                    
                    break  # Found surveys with this selector
            
            logger.info(f"Found {len(surveys)} available surveys on Swagbucks")
            return surveys
            
        except Exception as e:
            logger.error(f"Error getting Swagbucks surveys: {e}")
            return []
    
    async def start_survey(self, survey_id: str) -> bool:
        """Start a specific survey on Swagbucks."""
        try:
            # If survey_id is a URL, navigate to it
            if survey_id.startswith("http"):
                await self.browser.navigate_to(survey_id)
            else:
                # Try to find and click survey by ID or title
                survey_link = await self.browser.page.query_selector(f'a[href*="{survey_id}"]')
                if survey_link:
                    await self.browser.human_like_click(f'a[href*="{survey_id}"]')
                else:
                    logger.error(f"Could not find survey with ID: {survey_id}")
                    return False
            
            # Wait for survey to load
            await asyncio.sleep(3)
            
            # Handle any CAPTCHAs that might appear
            if not await self.handle_captcha_if_present():
                return False
            
            # Look for survey start indicators
            survey_indicators = [
                '.survey-content',
                '.question',
                'form[action*="survey"]',
                '.survey-question'
            ]
            
            for indicator in survey_indicators:
                if await self.browser.wait_for_element(indicator, timeout=10000):
                    logger.info(f"Successfully started survey: {survey_id}")
                    return True
            
            logger.warning("Survey page loaded but no survey content detected")
            return True
            
        except Exception as e:
            logger.error(f"Error starting Swagbucks survey: {e}")
            return False

class InboxDollarsHandler(BasePlatformHandler):
    """Handler for InboxDollars platform."""
    
    def __init__(self, browser_manager: BrowserManager, captcha_solver: CaptchaSolver):
        super().__init__(browser_manager, captcha_solver)
        self.platform_name = "InboxDollars"
        self.base_url = "https://www.inboxdollars.com"
        self.login_url = "https://www.inboxdollars.com/login"
        self.surveys_url = "https://www.inboxdollars.com/surveys"
        
    async def login(self, credentials: Dict[str, str]) -> bool:
        """Login to InboxDollars."""
        try:
            email = credentials.get("email")
            password = credentials.get("password")
            
            if not email or not password:
                logger.error("Email and password required for InboxDollars login")
                return False
            
            # Navigate to login page
            await self.browser.navigate_to(self.login_url)
            
            # Handle CAPTCHA if present
            if not await self.handle_captcha_if_present():
                return False
            
            # Fill login form
            email_selector = 'input[name="email"], input[id="email"], input[type="email"]'
            password_selector = 'input[name="password"], input[id="password"], input[type="password"]'
            
            # Wait for form elements
            if not await self.browser.wait_for_element(email_selector, timeout=10000):
                logger.error("Could not find email input field")
                return False
            
            # Fill credentials
            await self.browser.human_like_type(email_selector, email)
            await asyncio.sleep(1)
            await self.browser.human_like_type(password_selector, password)
            
            # Handle CAPTCHA again if it appeared after typing
            if not await self.handle_captcha_if_present():
                return False
            
            # Submit form
            submit_selectors = [
                'button[type="submit"]',
                'input[type="submit"]',
                'button:contains("Log In")',
                'button:contains("Sign In")',
                '.login-btn',
                '#loginBtn'
            ]
            
            submitted = False
            for selector in submit_selectors:
                try:
                    if await self.browser.wait_for_element(selector, timeout=2000):
                        await self.browser.human_like_click(selector)
                        submitted = True
                        break
                except:
                    continue
            
            if not submitted:
                # Try pressing Enter on password field
                await self.browser.page.press(password_selector, "Enter")
            
            # Wait for login to complete
            await asyncio.sleep(3)
            
            # Check if login was successful
            current_url = self.browser.page.url
            if "login" not in current_url.lower() and "inboxdollars.com" in current_url:
                # Look for user-specific elements
                user_indicators = [
                    '.user-menu',
                    '.member-name',
                    '.account-info',
                    '.dashboard'
                ]
                
                for indicator in user_indicators:
                    if await self.browser.wait_for_element(indicator, timeout=5000):
                        self.is_logged_in = True
                        logger.info("Successfully logged into InboxDollars")
                        return True
            
            logger.error("InboxDollars login failed - still on login page or no user indicators found")
            return False
            
        except Exception as e:
            logger.error(f"Error during InboxDollars login: {e}")
            return False
    
    async def navigate_to_surveys(self) -> bool:
        """Navigate to InboxDollars surveys section."""
        try:
            if not self.is_logged_in:
                logger.error("Must be logged in to navigate to surveys")
                return False
            
            # Navigate to surveys page
            await self.browser.navigate_to(self.surveys_url)
            
            # Wait for surveys to load
            survey_indicators = [
                '.survey-offer',
                '.survey-item',
                '[data-survey]',
                '.available-survey'
            ]
            
            for indicator in survey_indicators:
                if await self.browser.wait_for_element(indicator, timeout=10000):
                    logger.info("Successfully navigated to InboxDollars surveys")
                    return True
            
            logger.warning("Navigated to surveys page but no survey elements found")
            return True  # Page loaded, even if no surveys visible
            
        except Exception as e:
            logger.error(f"Error navigating to InboxDollars surveys: {e}")
            return False
    
    async def get_available_surveys(self) -> List[Dict[str, Any]]:
        """Get list of available surveys from InboxDollars."""
        try:
            surveys = []
            
            # Look for survey elements
            survey_selectors = [
                '.survey-offer',
                '.survey-item',
                '[data-survey]',
                '.offer-card'
            ]
            
            for selector in survey_selectors:
                survey_elements = await self.browser.page.query_selector_all(selector)
                if survey_elements:
                    for element in survey_elements:
                        try:
                            # Extract survey information
                            title_elem = await element.query_selector('.title, .survey-title, h3, h4')
                            reward_elem = await element.query_selector('.reward, .payout, .cash-amount')
                            time_elem = await element.query_selector('.time, .duration, .minutes')
                            
                            title = await title_elem.inner_text() if title_elem else "Unknown Survey"
                            reward = await reward_elem.inner_text() if reward_elem else "Unknown Reward"
                            time_est = await time_elem.inner_text() if time_elem else "Unknown Time"
                            
                            # Get survey link or ID
                            link_elem = await element.query_selector('a')
                            survey_url = await link_elem.get_attribute('href') if link_elem else None
                            
                            survey_data = {
                                "title": title.strip(),
                                "reward": reward.strip(),
                                "estimated_time": time_est.strip(),
                                "url": survey_url,
                                "platform": "InboxDollars"
                            }
                            
                            surveys.append(survey_data)
                            
                        except Exception as e:
                            logger.warning(f"Error extracting survey data: {e}")
                            continue
                    
                    break  # Found surveys with this selector
            
            logger.info(f"Found {len(surveys)} available surveys on InboxDollars")
            return surveys
            
        except Exception as e:
            logger.error(f"Error getting InboxDollars surveys: {e}")
            return []
    
    async def start_survey(self, survey_id: str) -> bool:
        """Start a specific survey on InboxDollars."""
        try:
            # If survey_id is a URL, navigate to it
            if survey_id.startswith("http"):
                await self.browser.navigate_to(survey_id)
            else:
                # Try to find and click survey by ID or title
                survey_link = await self.browser.page.query_selector(f'a[href*="{survey_id}"]')
                if survey_link:
                    await self.browser.human_like_click(f'a[href*="{survey_id}"]')
                else:
                    logger.error(f"Could not find survey with ID: {survey_id}")
                    return False
            
            # Wait for survey to load
            await asyncio.sleep(3)
            
            # Handle any CAPTCHAs that might appear
            if not await self.handle_captcha_if_present():
                return False
            
            # Look for survey start indicators
            survey_indicators = [
                '.survey-content',
                '.question',
                'form[action*="survey"]',
                '.survey-question'
            ]
            
            for indicator in survey_indicators:
                if await self.browser.wait_for_element(indicator, timeout=10000):
                    logger.info(f"Successfully started survey: {survey_id}")
                    return True
            
            logger.warning("Survey page loaded but no survey content detected")
            return True
            
        except Exception as e:
            logger.error(f"Error starting InboxDollars survey: {e}")
            return False

def get_platform_handler(platform: str, browser_manager: BrowserManager, captcha_solver: CaptchaSolver) -> Optional[BasePlatformHandler]:
    """Factory function to get the appropriate platform handler.
    
    Args:
        platform: Platform name ("swagbucks" or "inboxdollars")
        browser_manager: Browser manager instance
        captcha_solver: CAPTCHA solver instance
        
    Returns:
        Platform handler instance or None if platform not supported
    """
    platform = platform.lower()
    
    if platform == "swagbucks":
        return SwagbucksHandler(browser_manager, captcha_solver)
    elif platform == "inboxdollars":
        return InboxDollarsHandler(browser_manager, captcha_solver)
    else:
        logger.error(f"Unsupported platform: {platform}")
        return None

