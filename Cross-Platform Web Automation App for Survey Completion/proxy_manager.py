"""
Proxy Manager for IP Rotation
Handles proxy rotation and management for anti-detection.
"""

import random
import requests
import logging
from typing import List, Dict, Optional, Any
import time

logger = logging.getLogger(__name__)

class ProxyManager:
    """Manages proxy rotation for web automation."""
    
    def __init__(self, proxy_list: Optional[List[Dict[str, str]]] = None):
        self.proxy_list = proxy_list or []
        self.current_proxy_index = 0
        self.failed_proxies = set()
        self.proxy_test_timeout = 10
        
    def add_proxy(self, proxy_config: Dict[str, str]) -> None:
        """Add a proxy configuration to the list.
        
        Args:
            proxy_config: Dictionary with keys 'server', 'username', 'password'
        """
        self.proxy_list.append(proxy_config)
        logger.info(f"Added proxy: {proxy_config.get('server', 'Unknown')}")
    
    def add_proxies_from_list(self, proxies: List[str], username: str = None, password: str = None) -> None:
        """Add multiple proxies from a list of server addresses.
        
        Args:
            proxies: List of proxy server addresses (e.g., ['proxy1.com:8080', 'proxy2.com:8080'])
            username: Optional username for authentication
            password: Optional password for authentication
        """
        for proxy_server in proxies:
            proxy_config = {"server": proxy_server}
            if username and password:
                proxy_config.update({"username": username, "password": password})
            self.add_proxy(proxy_config)
    
    def get_current_proxy(self) -> Optional[Dict[str, str]]:
        """Get the current proxy configuration."""
        if not self.proxy_list:
            return None
        
        # Skip failed proxies
        attempts = 0
        while attempts < len(self.proxy_list):
            proxy = self.proxy_list[self.current_proxy_index]
            proxy_id = f"{proxy.get('server', '')}"
            
            if proxy_id not in self.failed_proxies:
                return proxy
            
            self.current_proxy_index = (self.current_proxy_index + 1) % len(self.proxy_list)
            attempts += 1
        
        # If all proxies failed, reset failed list and try again
        if attempts >= len(self.proxy_list):
            logger.warning("All proxies marked as failed, resetting failed list")
            self.failed_proxies.clear()
            return self.proxy_list[self.current_proxy_index] if self.proxy_list else None
        
        return None
    
    def rotate_proxy(self) -> Optional[Dict[str, str]]:
        """Rotate to the next proxy in the list."""
        if not self.proxy_list:
            return None
        
        self.current_proxy_index = (self.current_proxy_index + 1) % len(self.proxy_list)
        return self.get_current_proxy()
    
    def get_random_proxy(self) -> Optional[Dict[str, str]]:
        """Get a random proxy from the list."""
        if not self.proxy_list:
            return None
        
        available_proxies = [
            proxy for proxy in self.proxy_list
            if f"{proxy.get('server', '')}" not in self.failed_proxies
        ]
        
        if not available_proxies:
            # Reset failed proxies if none available
            self.failed_proxies.clear()
            available_proxies = self.proxy_list
        
        if available_proxies:
            proxy = random.choice(available_proxies)
            self.current_proxy_index = self.proxy_list.index(proxy)
            return proxy
        
        return None
    
    def test_proxy(self, proxy_config: Dict[str, str], test_url: str = "http://httpbin.org/ip") -> bool:
        """Test if a proxy is working.
        
        Args:
            proxy_config: Proxy configuration dictionary
            test_url: URL to test the proxy against
            
        Returns:
            True if proxy is working, False otherwise
        """
        try:
            proxy_server = proxy_config.get("server")
            if not proxy_server:
                return False
            
            # Prepare proxy for requests
            proxy_dict = {
                "http": f"http://{proxy_server}",
                "https": f"http://{proxy_server}"
            }
            
            # Add authentication if provided
            username = proxy_config.get("username")
            password = proxy_config.get("password")
            if username and password:
                proxy_dict = {
                    "http": f"http://{username}:{password}@{proxy_server}",
                    "https": f"http://{username}:{password}@{proxy_server}"
                }
            
            # Test the proxy
            response = requests.get(
                test_url,
                proxies=proxy_dict,
                timeout=self.proxy_test_timeout,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            )
            
            if response.status_code == 200:
                logger.info(f"Proxy {proxy_server} is working")
                return True
            else:
                logger.warning(f"Proxy {proxy_server} returned status code: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Proxy {proxy_config.get('server', 'Unknown')} failed test: {e}")
            return False
    
    def mark_proxy_failed(self, proxy_config: Dict[str, str]) -> None:
        """Mark a proxy as failed."""
        proxy_id = f"{proxy_config.get('server', '')}"
        self.failed_proxies.add(proxy_id)
        logger.warning(f"Marked proxy as failed: {proxy_id}")
    
    def test_all_proxies(self) -> List[Dict[str, str]]:
        """Test all proxies and return working ones."""
        working_proxies = []
        
        for proxy in self.proxy_list:
            if self.test_proxy(proxy):
                working_proxies.append(proxy)
            else:
                self.mark_proxy_failed(proxy)
        
        logger.info(f"Found {len(working_proxies)} working proxies out of {len(self.proxy_list)}")
        return working_proxies
    
    def get_proxy_stats(self) -> Dict[str, Any]:
        """Get statistics about proxy usage."""
        total_proxies = len(self.proxy_list)
        failed_proxies = len(self.failed_proxies)
        working_proxies = total_proxies - failed_proxies
        
        return {
            "total_proxies": total_proxies,
            "working_proxies": working_proxies,
            "failed_proxies": failed_proxies,
            "current_proxy": self.get_current_proxy(),
            "success_rate": (working_proxies / total_proxies * 100) if total_proxies > 0 else 0
        }
    
    def reset_failed_proxies(self) -> None:
        """Reset the failed proxies list."""
        self.failed_proxies.clear()
        logger.info("Reset failed proxies list")
    
    @staticmethod
    def create_free_proxy_list() -> List[str]:
        """Create a list of free proxy servers (for testing purposes only).
        
        Note: Free proxies are unreliable and should not be used in production.
        """
        # This is just an example - in production, use paid proxy services
        free_proxies = [
            "proxy1.example.com:8080",
            "proxy2.example.com:8080",
            "proxy3.example.com:8080"
        ]
        
        logger.warning("Using free proxies - not recommended for production use")
        return free_proxies
    
    def configure_for_playwright(self, proxy_config: Dict[str, str]) -> Dict[str, str]:
        """Convert proxy config to Playwright format.
        
        Args:
            proxy_config: Internal proxy configuration
            
        Returns:
            Playwright-compatible proxy configuration
        """
        if not proxy_config or not proxy_config.get("server"):
            return {}
        
        playwright_config = {
            "server": f"http://{proxy_config['server']}"
        }
        
        if proxy_config.get("username") and proxy_config.get("password"):
            playwright_config.update({
                "username": proxy_config["username"],
                "password": proxy_config["password"]
            })
        
        return playwright_config

