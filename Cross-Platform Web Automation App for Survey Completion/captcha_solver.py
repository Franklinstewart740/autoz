"""
CAPTCHA Solver Module
Handles various types of CAPTCHAs using OCR and third-party services.
"""

import asyncio
import base64
import io
import logging
import os
import time
from typing import Optional, Dict, Any, Tuple
import requests
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class CaptchaSolver:
    """Handles CAPTCHA detection and solving."""
    
    def __init__(self, api_key: Optional[str] = None, service: str = "2captcha"):
        self.api_key = api_key
        self.service = service
        self.base_urls = {
            "2captcha": "http://2captcha.com",
            "anticaptcha": "https://api.anti-captcha.com"
        }
        
    async def detect_captcha_type(self, page) -> Optional[str]:
        """Detect the type of CAPTCHA on the page.
        
        Returns:
            String indicating CAPTCHA type or None if no CAPTCHA found
        """
        try:
            # Check for reCAPTCHA v2
            recaptcha_v2 = await page.query_selector('iframe[src*="recaptcha"]')
            if recaptcha_v2:
                return "recaptcha_v2"
            
            # Check for reCAPTCHA v3 (invisible)
            recaptcha_v3_script = await page.query_selector('script[src*="recaptcha/api.js"]')
            if recaptcha_v3_script:
                return "recaptcha_v3"
            
            # Check for hCaptcha
            hcaptcha = await page.query_selector('iframe[src*="hcaptcha"]')
            if hcaptcha:
                return "hcaptcha"
            
            # Check for image CAPTCHA
            captcha_images = await page.query_selector_all('img[src*="captcha"], img[alt*="captcha"], img[id*="captcha"]')
            if captcha_images:
                return "image_captcha"
            
            # Check for text input CAPTCHAs
            captcha_inputs = await page.query_selector_all('input[name*="captcha"], input[id*="captcha"]')
            if captcha_inputs:
                return "text_captcha"
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting CAPTCHA type: {e}")
            return None
    
    async def solve_captcha(self, page, captcha_type: str) -> Optional[str]:
        """Solve CAPTCHA based on its type.
        
        Args:
            page: Playwright page object
            captcha_type: Type of CAPTCHA to solve
            
        Returns:
            CAPTCHA solution or None if failed
        """
        try:
            if captcha_type == "recaptcha_v2":
                return await self._solve_recaptcha_v2(page)
            elif captcha_type == "recaptcha_v3":
                return await self._solve_recaptcha_v3(page)
            elif captcha_type == "hcaptcha":
                return await self._solve_hcaptcha(page)
            elif captcha_type == "image_captcha":
                return await self._solve_image_captcha(page)
            elif captcha_type == "text_captcha":
                return await self._solve_text_captcha(page)
            else:
                logger.warning(f"Unsupported CAPTCHA type: {captcha_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error solving CAPTCHA: {e}")
            return None
    
    async def _solve_recaptcha_v2(self, page) -> Optional[str]:
        """Solve reCAPTCHA v2 using third-party service."""
        if not self.api_key:
            logger.error("API key required for reCAPTCHA v2 solving")
            return None
        
        try:
            # Get site key
            site_key = await page.evaluate("""
                () => {
                    const recaptcha = document.querySelector('[data-sitekey]');
                    return recaptcha ? recaptcha.getAttribute('data-sitekey') : null;
                }
            """)
            
            if not site_key:
                logger.error("Could not find reCAPTCHA site key")
                return None
            
            current_url = page.url
            
            # Submit CAPTCHA to solving service
            if self.service == "2captcha":
                return await self._solve_with_2captcha(
                    method="userrecaptcha",
                    googlekey=site_key,
                    pageurl=current_url
                )
            
        except Exception as e:
            logger.error(f"Error solving reCAPTCHA v2: {e}")
            return None
    
    async def _solve_recaptcha_v3(self, page) -> Optional[str]:
        """Solve reCAPTCHA v3 using third-party service."""
        if not self.api_key:
            logger.error("API key required for reCAPTCHA v3 solving")
            return None
        
        try:
            # Get site key and action
            site_key = await page.evaluate("""
                () => {
                    const scripts = document.querySelectorAll('script');
                    for (let script of scripts) {
                        const content = script.textContent || script.innerText;
                        const match = content.match(/sitekey['"\\s]*[:=]['"\\s]*([^'"\\s]+)/);
                        if (match) return match[1];
                    }
                    return null;
                }
            """)
            
            if not site_key:
                logger.error("Could not find reCAPTCHA v3 site key")
                return None
            
            current_url = page.url
            
            # Submit CAPTCHA to solving service
            if self.service == "2captcha":
                return await self._solve_with_2captcha(
                    method="userrecaptcha",
                    googlekey=site_key,
                    pageurl=current_url,
                    version="v3",
                    action="submit",
                    min_score=0.3
                )
            
        except Exception as e:
            logger.error(f"Error solving reCAPTCHA v3: {e}")
            return None
    
    async def _solve_hcaptcha(self, page) -> Optional[str]:
        """Solve hCaptcha using third-party service."""
        if not self.api_key:
            logger.error("API key required for hCaptcha solving")
            return None
        
        try:
            # Get site key
            site_key = await page.evaluate("""
                () => {
                    const hcaptcha = document.querySelector('[data-sitekey]');
                    return hcaptcha ? hcaptcha.getAttribute('data-sitekey') : null;
                }
            """)
            
            if not site_key:
                logger.error("Could not find hCaptcha site key")
                return None
            
            current_url = page.url
            
            # Submit CAPTCHA to solving service
            if self.service == "2captcha":
                return await self._solve_with_2captcha(
                    method="hcaptcha",
                    sitekey=site_key,
                    pageurl=current_url
                )
            
        except Exception as e:
            logger.error(f"Error solving hCaptcha: {e}")
            return None
    
    async def _solve_image_captcha(self, page) -> Optional[str]:
        """Solve image CAPTCHA using OCR or third-party service."""
        try:
            # Find CAPTCHA image
            captcha_img = await page.query_selector('img[src*="captcha"], img[alt*="captcha"], img[id*="captcha"]')
            if not captcha_img:
                logger.error("Could not find CAPTCHA image")
                return None
            
            # Get image data
            img_data = await captcha_img.screenshot()
            
            # Try OCR first (for simple CAPTCHAs)
            ocr_result = await self._solve_with_ocr(img_data)
            if ocr_result and len(ocr_result) > 2:  # Basic validation
                return ocr_result
            
            # Fall back to third-party service
            if self.api_key:
                img_base64 = base64.b64encode(img_data).decode('utf-8')
                return await self._solve_with_2captcha(
                    method="base64",
                    body=img_base64
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error solving image CAPTCHA: {e}")
            return None
    
    async def _solve_text_captcha(self, page) -> Optional[str]:
        """Solve text-based CAPTCHA using OCR."""
        try:
            # This is similar to image CAPTCHA but specifically for text
            return await self._solve_image_captcha(page)
            
        except Exception as e:
            logger.error(f"Error solving text CAPTCHA: {e}")
            return None
    
    async def _solve_with_ocr(self, image_data: bytes) -> Optional[str]:
        """Solve CAPTCHA using OCR (Tesseract).
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Extracted text or None if failed
        """
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # Preprocess image for better OCR
            processed_image = self._preprocess_image(image)
            
            # Use Tesseract OCR (would need pytesseract installed)
            # For now, return a placeholder
            # import pytesseract
            # text = pytesseract.image_to_string(processed_image, config='--psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
            # return text.strip()
            
            logger.info("OCR solving not implemented - would use Tesseract here")
            return None
            
        except Exception as e:
            logger.error(f"OCR solving failed: {e}")
            return None
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR accuracy."""
        try:
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            
            # Apply filters to reduce noise
            image = image.filter(ImageFilter.MedianFilter())
            
            # Resize if too small
            width, height = image.size
            if width < 100 or height < 30:
                scale_factor = max(100 / width, 30 / height)
                new_size = (int(width * scale_factor), int(height * scale_factor))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            return image
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return image
    
    async def _solve_with_2captcha(self, **kwargs) -> Optional[str]:
        """Solve CAPTCHA using 2captcha service."""
        if not self.api_key:
            return None
        
        try:
            # Submit CAPTCHA
            submit_url = f"{self.base_urls['2captcha']}/in.php"
            submit_data = {
                "key": self.api_key,
                "json": 1,
                **kwargs
            }
            
            response = requests.post(submit_url, data=submit_data, timeout=30)
            result = response.json()
            
            if result.get("status") != 1:
                logger.error(f"2captcha submission failed: {result.get('error_text', 'Unknown error')}")
                return None
            
            captcha_id = result.get("request")
            if not captcha_id:
                logger.error("No CAPTCHA ID received from 2captcha")
                return None
            
            # Poll for result
            result_url = f"{self.base_urls['2captcha']}/res.php"
            max_attempts = 30  # 5 minutes with 10-second intervals
            
            for attempt in range(max_attempts):
                await asyncio.sleep(10)  # Wait 10 seconds between checks
                
                result_data = {
                    "key": self.api_key,
                    "action": "get",
                    "id": captcha_id,
                    "json": 1
                }
                
                response = requests.get(result_url, params=result_data, timeout=30)
                result = response.json()
                
                if result.get("status") == 1:
                    solution = result.get("request")
                    logger.info("CAPTCHA solved successfully with 2captcha")
                    return solution
                elif result.get("error_text") == "CAPCHA_NOT_READY":
                    continue
                else:
                    logger.error(f"2captcha solving failed: {result.get('error_text', 'Unknown error')}")
                    return None
            
            logger.error("2captcha solving timed out")
            return None
            
        except Exception as e:
            logger.error(f"Error with 2captcha service: {e}")
            return None
    
    async def submit_captcha_solution(self, page, captcha_type: str, solution: str) -> bool:
        """Submit CAPTCHA solution to the page.
        
        Args:
            page: Playwright page object
            captcha_type: Type of CAPTCHA
            solution: CAPTCHA solution
            
        Returns:
            True if submission successful, False otherwise
        """
        try:
            if captcha_type in ["recaptcha_v2", "recaptcha_v3"]:
                # Inject reCAPTCHA solution
                await page.evaluate(f"""
                    () => {{
                        if (window.grecaptcha && window.grecaptcha.getResponse) {{
                            document.getElementById('g-recaptcha-response').innerHTML = '{solution}';
                            document.getElementById('g-recaptcha-response').style.display = 'block';
                        }}
                    }}
                """)
                
            elif captcha_type == "hcaptcha":
                # Inject hCaptcha solution
                await page.evaluate(f"""
                    () => {{
                        const response = document.querySelector('[name="h-captcha-response"]');
                        if (response) {{
                            response.value = '{solution}';
                        }}
                    }}
                """)
                
            elif captcha_type in ["image_captcha", "text_captcha"]:
                # Find and fill CAPTCHA input
                captcha_input = await page.query_selector('input[name*="captcha"], input[id*="captcha"]')
                if captcha_input:
                    await captcha_input.fill(solution)
                else:
                    logger.error("Could not find CAPTCHA input field")
                    return False
            
            logger.info(f"CAPTCHA solution submitted for {captcha_type}")
            return True
            
        except Exception as e:
            logger.error(f"Error submitting CAPTCHA solution: {e}")
            return False

