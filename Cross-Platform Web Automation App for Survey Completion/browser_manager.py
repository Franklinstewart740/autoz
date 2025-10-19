"""
Browser Manager with Anti-Detection Features
Handles browser automation with stealth capabilities and proxy rotation.
"""

import asyncio
import random
import time
from typing import Optional, Dict, Any, List
from playwright.async_api import async_playwright, Browser, BrowserContext, Page
from playwright_stealth import stealth
import logging

logger = logging.getLogger(__name__)

class BrowserManager:
    """Manages browser instances with anti-detection capabilities."""
    
    def __init__(self, proxy_config: Optional[Dict[str, str]] = None):
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.proxy_config = proxy_config
        
    async def start_browser(self, headless: bool = True, browser_type: str = "chromium") -> None:
        """Start browser with anti-detection features."""
        try:
            self.playwright = await async_playwright().start()
            
            # Browser launch options with anti-detection
            launch_options = {
                "headless": headless,
                "args": [
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-accelerated-2d-canvas",
                    "--no-first-run",
                    "--no-zygote",
                    "--disable-gpu",
                    "--disable-background-timer-throttling",
                    "--disable-backgrounding-occluded-windows",
                    "--disable-renderer-backgrounding",
                    "--disable-features=TranslateUI",
                    "--disable-ipc-flooding-protection",
                    "--disable-web-security",
                    "--disable-features=VizDisplayCompositor"
                ]
            }
            
            # Add proxy if configured
            if self.proxy_config:
                launch_options["proxy"] = self.proxy_config
            
            # Launch browser
            if browser_type == "chromium":
                self.browser = await self.playwright.chromium.launch(**launch_options)
            elif browser_type == "firefox":
                self.browser = await self.playwright.firefox.launch(**launch_options)
            elif browser_type == "webkit":
                self.browser = await self.playwright.webkit.launch(**launch_options)
            else:
                raise ValueError(f"Unsupported browser type: {browser_type}")
            
            # Create context with anti-detection settings
            context_options = {
                "viewport": {"width": 1920, "height": 1080},
                "user_agent": self._get_random_user_agent(),
                "locale": "en-US",
                "timezone_id": "America/New_York",
                "permissions": ["geolocation"],
                "geolocation": {"latitude": 40.7128, "longitude": -74.0060},
                "extra_http_headers": {
                    "Accept-Language": "en-US,en;q=0.9",
                    "Accept-Encoding": "gzip, deflate, br",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1"
                }
            }
            
            self.context = await self.browser.new_context(**context_options)
            
            # Create page and apply stealth
            self.page = await self.context.new_page()
            await stealth(self.page)
            
            # Override navigator properties to avoid detection
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
            
            logger.info(f"Browser started successfully with {browser_type}")
            
        except Exception as e:
            logger.error(f"Failed to start browser: {e}")
            await self.close()
            raise
    
    def _get_random_user_agent(self) -> str:
        """Get a random user agent string."""
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ]
        return random.choice(user_agents)
    
    async def navigate_to(self, url: str, wait_for_load: bool = True) -> None:
        """Navigate to a URL with human-like behavior."""
        if not self.page:
            raise RuntimeError("Browser not started. Call start_browser() first.")
        
        try:
            # Add random delay before navigation
            await asyncio.sleep(random.uniform(1, 3))
            
            # Navigate to URL
            await self.page.goto(url, wait_until="networkidle", timeout=30000)
            
            if wait_for_load:
                # Wait for page to be fully loaded
                await self.page.wait_for_load_state("domcontentloaded")
                await asyncio.sleep(random.uniform(2, 4))
            
            logger.info(f"Navigated to: {url}")
            
        except Exception as e:
            logger.error(f"Failed to navigate to {url}: {e}")
            raise
    
    async def human_like_click(self, selector: str, delay_range: tuple = (0.5, 2.0)) -> None:
        """Click an element with human-like behavior."""
        if not self.page:
            raise RuntimeError("Browser not started.")
        
        try:
            # Wait for element to be visible
            await self.page.wait_for_selector(selector, state="visible", timeout=10000)
            
            # Add random delay
            await asyncio.sleep(random.uniform(*delay_range))
            
            # Move mouse to element first
            element = await self.page.query_selector(selector)
            if element:
                box = await element.bounding_box()
                if box:
                    # Add slight randomness to click position
                    x = box["x"] + box["width"] / 2 + random.uniform(-5, 5)
                    y = box["y"] + box["height"] / 2 + random.uniform(-5, 5)
                    
                    await self.page.mouse.move(x, y)
                    await asyncio.sleep(random.uniform(0.1, 0.3))
                    await self.page.mouse.click(x, y)
            else:
                await self.page.click(selector)
            
            logger.info(f"Clicked element: {selector}")
            
        except Exception as e:
            logger.error(f"Failed to click {selector}: {e}")
            raise
    
    async def human_like_type(self, selector: str, text: str, delay_range: tuple = (0.05, 0.15)) -> None:
        """Type text with human-like behavior."""
        if not self.page:
            raise RuntimeError("Browser not started.")
        
        try:
            # Wait for element and focus
            await self.page.wait_for_selector(selector, state="visible", timeout=10000)
            await self.page.focus(selector)
            
            # Clear existing text
            await self.page.fill(selector, "")
            
            # Type with random delays between characters
            for char in text:
                await self.page.type(selector, char, delay=random.uniform(*delay_range) * 1000)
            
            logger.info(f"Typed text into: {selector}")
            
        except Exception as e:
            logger.error(f"Failed to type into {selector}: {e}")
            raise
    
    async def wait_for_element(self, selector: str, timeout: int = 10000) -> bool:
        """Wait for an element to appear."""
        if not self.page:
            raise RuntimeError("Browser not started.")
        
        try:
            await self.page.wait_for_selector(selector, timeout=timeout)
            return True
        except Exception:
            return False
    
    async def get_page_content(self) -> str:
        """Get the current page content."""
        if not self.page:
            raise RuntimeError("Browser not started.")
        
        return await self.page.content()
    
    async def take_screenshot(self, path: str) -> None:
        """Take a screenshot of the current page."""
        if not self.page:
            raise RuntimeError("Browser not started.")
        
        await self.page.screenshot(path=path, full_page=True)
        logger.info(f"Screenshot saved to: {path}")
    
    async def close(self) -> None:
        """Close browser and cleanup resources."""
        try:
            if self.page:
                await self.page.close()
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
            
            logger.info("Browser closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing browser: {e}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

