"""
Navigator Agent
Specialized agent responsible for web browsing, navigation, and platform interaction.
Handles login, survey discovery, and general web interactions.
"""

import asyncio
import random
import time
from typing import Any, Dict, List, Optional
from playwright.async_api import async_playwright, Browser, BrowserContext, Page
from playwright_stealth import stealth

from .base_agent import BaseAgent, AgentTask


class NavigatorAgent(BaseAgent):
    """
    Navigator Agent handles all web browsing and navigation tasks.
    Specialized in:
    - Browser management with anti-detection
    - Platform login and authentication
    - Survey discovery and navigation
    - Page interaction and element location
    """

    def __init__(self, agent_id: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(agent_id, "navigator", config)
        
        # Browser management
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        
        # Navigation state
        self.current_url = ""
        self.login_status = {}
        self.platform_sessions = {}
        
        # Performance tracking
        self.page_load_times = []
        self.interaction_delays = []

    async def initialize(self) -> bool:
        """Initialize the Navigator Agent with browser setup."""
        try:
            self.logger.info("Initializing Navigator Agent")
            
            # Start Playwright
            self.playwright = await async_playwright().start()
            
            # Configure browser options
            browser_options = await self._get_browser_options()
            
            # Launch browser
            self.browser = await self.playwright.chromium.launch(**browser_options)
            
            # Create context with anti-detection
            context_options = await self._get_context_options()
            self.context = await self.browser.new_context(**context_options)
            
            # Create page
            self.page = await self.context.new_page()
            
            # Apply stealth modifications
            await stealth(self.page)
            
            # Apply basic anti-detection measures
            await self._apply_basic_anti_detection()
            
            self.logger.info("Navigator Agent initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Navigator Agent: {e}", exc_info=True)
            return False

    async def cleanup(self) -> None:
        """Clean up browser resources."""
        try:
            if self.page:
                await self.page.close()
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
            
            self.logger.info("Navigator Agent cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during Navigator Agent cleanup: {e}", exc_info=True)

    async def process_task(self, task: AgentTask) -> Dict[str, Any]:
        """
        Process navigation-related tasks.
        
        Args:
            task: The navigation task to process
            
        Returns:
            Dictionary containing the task result
        """
        task_type = task.task_type
        payload = task.payload
        
        self.logger.info(f"Processing navigation task: {task_type}")
        
        try:
            if task_type == "navigate_to_url":
                return await self._navigate_to_url(payload)
            
            elif task_type == "login_to_platform":
                return await self._login_to_platform(payload)
            
            elif task_type == "discover_surveys":
                return await self._discover_surveys(payload)
            
            elif task_type == "navigate_to_survey":
                return await self._navigate_to_survey(payload)
            
            elif task_type == "interact_with_element":
                return await self._interact_with_element(payload)
            
            elif task_type == "wait_for_element":
                return await self._wait_for_element(payload)
            
            elif task_type == "get_page_info":
                return await self._get_page_info(payload)
            
            elif task_type == "handle_popup":
                return await self._handle_popup(payload)
            
            elif task_type == "scroll_page":
                return await self._scroll_page(payload)
            
            else:
                raise ValueError(f"Unknown navigation task type: {task_type}")
                
        except Exception as e:
            self.logger.error(f"Error processing navigation task {task_type}: {e}", exc_info=True)
            raise

    async def _navigate_to_url(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Navigate to a specific URL."""
        url = payload["url"]
        wait_until = payload.get("wait_until", "domcontentloaded")
        timeout = payload.get("timeout", 30000)
        
        start_time = time.time()
        
        try:
            # Add human-like delay before navigation
            await self._human_delay(0.5, 2.0)
            
            # Navigate to URL
            response = await self.page.goto(url, wait_until=wait_until, timeout=timeout)
            
            # Record page load time
            load_time = time.time() - start_time
            self.page_load_times.append(load_time)
            
            # Update current URL
            self.current_url = self.page.url
            
            # Wait for page to stabilize
            await self._wait_for_page_stability()
            
            return {
                "success": True,
                "url": self.current_url,
                "status_code": response.status if response else None,
                "load_time": load_time,
                "title": await self.page.title()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "url": url
            }

    async def _login_to_platform(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle platform login process."""
        platform = payload["platform"]
        credentials = payload["credentials"]
        
        self.logger.info(f"Logging into platform: {platform}")
        
        try:
            if platform.lower() == "swagbucks":
                return await self._login_swagbucks(credentials)
            elif platform.lower() == "inboxdollars":
                return await self._login_inboxdollars(credentials)
            else:
                raise ValueError(f"Unsupported platform: {platform}")
                
        except Exception as e:
            return {
                "success": False,
                "platform": platform,
                "error": str(e)
            }

    async def _login_swagbucks(self, credentials: Dict[str, str]) -> Dict[str, Any]:
        """Handle Swagbucks login."""
        try:
            # Navigate to Swagbucks login page
            await self.page.goto("https://www.swagbucks.com/login", wait_until="domcontentloaded")
            
            # Wait for login form
            await self.page.wait_for_selector('input[name="emailAddress"]', timeout=10000)
            
            # Fill credentials with human-like typing
            await self._type_like_human('input[name="emailAddress"]', credentials["email"])
            await self._human_delay(0.5, 1.5)
            
            await self._type_like_human('input[name="password"]', credentials["password"])
            await self._human_delay(0.5, 1.5)
            
            # Click login button
            await self.page.click('button[type="submit"]')
            
            # Wait for navigation or error
            try:
                await self.page.wait_for_url("**/dashboard**", timeout=15000)
                login_success = True
                error_message = None
            except:
                # Check for error messages
                error_element = await self.page.query_selector('.error-message, .alert-danger')
                error_message = await error_element.inner_text() if error_element else "Login failed"
                login_success = False
            
            # Update login status
            self.login_status["swagbucks"] = {
                "logged_in": login_success,
                "timestamp": time.time(),
                "session_id": await self._get_session_id()
            }
            
            return {
                "success": login_success,
                "platform": "swagbucks",
                "error": error_message,
                "current_url": self.page.url
            }
            
        except Exception as e:
            return {
                "success": False,
                "platform": "swagbucks",
                "error": str(e)
            }

    async def _login_inboxdollars(self, credentials: Dict[str, str]) -> Dict[str, Any]:
        """Handle InboxDollars login."""
        try:
            # Navigate to InboxDollars login page
            await self.page.goto("https://www.inboxdollars.com/login", wait_until="domcontentloaded")
            
            # Wait for login form
            await self.page.wait_for_selector('input[name="email"]', timeout=10000)
            
            # Fill credentials with human-like typing
            await self._type_like_human('input[name="email"]', credentials["email"])
            await self._human_delay(0.5, 1.5)
            
            await self._type_like_human('input[name="password"]', credentials["password"])
            await self._human_delay(0.5, 1.5)
            
            # Click login button
            await self.page.click('button[type="submit"]')
            
            # Wait for navigation or error
            try:
                await self.page.wait_for_url("**/member/**", timeout=15000)
                login_success = True
                error_message = None
            except:
                # Check for error messages
                error_element = await self.page.query_selector('.error, .alert')
                error_message = await error_element.inner_text() if error_element else "Login failed"
                login_success = False
            
            # Update login status
            self.login_status["inboxdollars"] = {
                "logged_in": login_success,
                "timestamp": time.time(),
                "session_id": await self._get_session_id()
            }
            
            return {
                "success": login_success,
                "platform": "inboxdollars",
                "error": error_message,
                "current_url": self.page.url
            }
            
        except Exception as e:
            return {
                "success": False,
                "platform": "inboxdollars",
                "error": str(e)
            }

    async def _discover_surveys(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Discover available surveys on the current platform."""
        platform = payload.get("platform", "").lower()
        
        try:
            if platform == "swagbucks":
                return await self._discover_swagbucks_surveys()
            elif platform == "inboxdollars":
                return await self._discover_inboxdollars_surveys()
            else:
                raise ValueError(f"Survey discovery not implemented for platform: {platform}")
                
        except Exception as e:
            return {
                "success": False,
                "surveys": [],
                "error": str(e)
            }

    async def _discover_swagbucks_surveys(self) -> Dict[str, Any]:
        """Discover Swagbucks surveys."""
        try:
            # Navigate to surveys page
            await self.page.goto("https://www.swagbucks.com/surveys", wait_until="domcontentloaded")
            
            # Wait for surveys to load
            await self.page.wait_for_selector('.survey-card, .survey-item', timeout=10000)
            
            # Extract survey information
            surveys = await self.page.evaluate("""
                () => {
                    const surveyElements = document.querySelectorAll('.survey-card, .survey-item');
                    return Array.from(surveyElements).map(element => {
                        const titleElement = element.querySelector('.survey-title, h3, .title');
                        const rewardElement = element.querySelector('.reward, .points, .sb-amount');
                        const timeElement = element.querySelector('.time, .duration, .minutes');
                        const linkElement = element.querySelector('a');
                        
                        return {
                            title: titleElement ? titleElement.textContent.trim() : '',
                            reward: rewardElement ? rewardElement.textContent.trim() : '',
                            time: timeElement ? timeElement.textContent.trim() : '',
                            url: linkElement ? linkElement.href : '',
                            id: linkElement ? linkElement.getAttribute('data-survey-id') || linkElement.href.split('/').pop() : ''
                        };
                    }).filter(survey => survey.title && survey.url);
                }
            """)
            
            return {
                "success": True,
                "surveys": surveys,
                "count": len(surveys),
                "platform": "swagbucks"
            }
            
        except Exception as e:
            return {
                "success": False,
                "surveys": [],
                "error": str(e),
                "platform": "swagbucks"
            }

    async def _discover_inboxdollars_surveys(self) -> Dict[str, Any]:
        """Discover InboxDollars surveys."""
        try:
            # Navigate to surveys page
            await self.page.goto("https://www.inboxdollars.com/online-surveys", wait_until="domcontentloaded")
            
            # Wait for surveys to load
            await self.page.wait_for_selector('.survey-offer, .survey-item', timeout=10000)
            
            # Extract survey information
            surveys = await self.page.evaluate("""
                () => {
                    const surveyElements = document.querySelectorAll('.survey-offer, .survey-item');
                    return Array.from(surveyElements).map(element => {
                        const titleElement = element.querySelector('.survey-title, h3, .title');
                        const rewardElement = element.querySelector('.reward, .amount, .cash');
                        const timeElement = element.querySelector('.time, .duration, .minutes');
                        const linkElement = element.querySelector('a');
                        
                        return {
                            title: titleElement ? titleElement.textContent.trim() : '',
                            reward: rewardElement ? rewardElement.textContent.trim() : '',
                            time: timeElement ? timeElement.textContent.trim() : '',
                            url: linkElement ? linkElement.href : '',
                            id: linkElement ? linkElement.getAttribute('data-survey-id') || linkElement.href.split('/').pop() : ''
                        };
                    }).filter(survey => survey.title && survey.url);
                }
            """)
            
            return {
                "success": True,
                "surveys": surveys,
                "count": len(surveys),
                "platform": "inboxdollars"
            }
            
        except Exception as e:
            return {
                "success": False,
                "surveys": [],
                "error": str(e),
                "platform": "inboxdollars"
            }

    async def _navigate_to_survey(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Navigate to a specific survey."""
        survey_url = payload["survey_url"]
        survey_id = payload.get("survey_id", "")
        
        try:
            # Navigate to survey
            result = await self._navigate_to_url({"url": survey_url})
            
            if result["success"]:
                # Wait for survey to load
                await self._wait_for_page_stability()
                
                # Check if survey is still available
                is_available = await self._check_survey_availability()
                
                return {
                    "success": True,
                    "survey_id": survey_id,
                    "survey_url": survey_url,
                    "current_url": self.page.url,
                    "available": is_available,
                    "title": await self.page.title()
                }
            else:
                return result
                
        except Exception as e:
            return {
                "success": False,
                "survey_id": survey_id,
                "error": str(e)
            }

    async def _interact_with_element(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Interact with a page element."""
        selector = payload["selector"]
        action = payload["action"]
        value = payload.get("value", "")
        
        try:
            # Wait for element to be available
            await self.page.wait_for_selector(selector, timeout=10000)
            
            # Add human-like delay
            await self._human_delay(0.3, 1.0)
            
            if action == "click":
                await self.page.click(selector)
            elif action == "type":
                await self._type_like_human(selector, value)
            elif action == "select":
                await self.page.select_option(selector, value)
            elif action == "check":
                await self.page.check(selector)
            elif action == "uncheck":
                await self.page.uncheck(selector)
            else:
                raise ValueError(f"Unknown action: {action}")
            
            return {
                "success": True,
                "selector": selector,
                "action": action,
                "value": value
            }
            
        except Exception as e:
            return {
                "success": False,
                "selector": selector,
                "action": action,
                "error": str(e)
            }

    async def _wait_for_element(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Wait for an element to appear or disappear."""
        selector = payload["selector"]
        state = payload.get("state", "visible")
        timeout = payload.get("timeout", 10000)
        
        try:
            if state == "visible":
                await self.page.wait_for_selector(selector, timeout=timeout)
            elif state == "hidden":
                await self.page.wait_for_selector(selector, state="hidden", timeout=timeout)
            elif state == "attached":
                await self.page.wait_for_selector(selector, state="attached", timeout=timeout)
            elif state == "detached":
                await self.page.wait_for_selector(selector, state="detached", timeout=timeout)
            
            return {
                "success": True,
                "selector": selector,
                "state": state
            }
            
        except Exception as e:
            return {
                "success": False,
                "selector": selector,
                "state": state,
                "error": str(e)
            }

    async def _get_page_info(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Get information about the current page."""
        try:
            info = {
                "url": self.page.url,
                "title": await self.page.title(),
                "content": await self.page.content() if payload.get("include_content") else None,
                "viewport": self.page.viewport_size,
                "cookies": await self.context.cookies(),
                "local_storage": await self.page.evaluate("() => Object.fromEntries(Object.entries(localStorage))")
            }
            
            return {
                "success": True,
                "page_info": info
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _handle_popup(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle popup dialogs."""
        action = payload.get("action", "accept")
        
        try:
            # Set up popup handler
            popup_handled = False
            
            def handle_dialog(dialog):
                nonlocal popup_handled
                if action == "accept":
                    dialog.accept()
                else:
                    dialog.dismiss()
                popup_handled = True
            
            self.page.on("dialog", handle_dialog)
            
            # Wait for popup or timeout
            await asyncio.sleep(5)
            
            return {
                "success": True,
                "popup_handled": popup_handled,
                "action": action
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _scroll_page(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Scroll the page."""
        direction = payload.get("direction", "down")
        amount = payload.get("amount", 500)
        
        try:
            if direction == "down":
                await self.page.evaluate(f"window.scrollBy(0, {amount})")
            elif direction == "up":
                await self.page.evaluate(f"window.scrollBy(0, -{amount})")
            elif direction == "top":
                await self.page.evaluate("window.scrollTo(0, 0)")
            elif direction == "bottom":
                await self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            
            # Add human-like delay after scrolling
            await self._human_delay(0.5, 1.5)
            
            return {
                "success": True,
                "direction": direction,
                "amount": amount
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    # Helper methods

    async def _get_browser_options(self) -> Dict[str, Any]:
        """Get browser launch options with anti-detection."""
        return {
            "headless": self.config.get("headless", True),
            "args": [
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
                "--disable-web-security",
                "--disable-features=VizDisplayCompositor",
                "--disable-dev-shm-usage",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-extensions-file-access-check",
                "--disable-extensions-http-throttling",
                "--disable-extensions-except",
                "--disable-component-extensions-with-background-pages"
            ]
        }

    async def _get_context_options(self) -> Dict[str, Any]:
        """Get browser context options."""
        return {
            "viewport": {"width": 1366, "height": 768},
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "locale": "en-US",
            "timezone_id": "America/New_York",
            "permissions": ["geolocation", "notifications"]
        }

    async def _type_like_human(self, selector: str, text: str) -> None:
        """Type text with human-like delays."""
        await self.page.focus(selector)
        
        for char in text:
            await self.page.keyboard.type(char)
            # Random delay between keystrokes
            await asyncio.sleep(random.uniform(0.05, 0.15))

    async def _human_delay(self, min_seconds: float = 0.5, max_seconds: float = 2.0) -> None:
        """Add a human-like delay."""
        delay = random.uniform(min_seconds, max_seconds)
        await asyncio.sleep(delay)
        self.interaction_delays.append(delay)

    async def _wait_for_page_stability(self) -> None:
        """Wait for the page to become stable (no network activity)."""
        try:
            await self.page.wait_for_load_state("networkidle", timeout=10000)
        except:
            # Fallback to a simple delay
            await asyncio.sleep(2)

    async def _check_survey_availability(self) -> bool:
        """Check if the current survey is still available."""
        try:
            # Look for common "survey not available" indicators
            unavailable_indicators = [
                "survey is no longer available",
                "survey has ended",
                "quota full",
                "not qualified",
                "survey closed"
            ]
            
            page_text = (await self.page.content()).lower()
            
            for indicator in unavailable_indicators:
                if indicator in page_text:
                    return False
            
            return True
            
        except:
            return True  # Assume available if we can't check

    async def _get_session_id(self) -> Optional[str]:
        """Extract session ID from cookies or page."""
        try:
            cookies = await self.context.cookies()
            for cookie in cookies:
                if "session" in cookie["name"].lower():
                    return cookie["value"]
            return None
        except:
            return None


    async def _apply_basic_anti_detection(self) -> None:
        """Apply basic anti-detection measures to the page."""
        await self.page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
            });
            
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5],
            });
            
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en'],
            });
            
            window.chrome = {
                runtime: {},
            };
            
            Object.defineProperty(navigator, 'permissions', {
                get: () => ({
                    query: async () => ({ state: 'granted' }),
                }),
            });
        """)

