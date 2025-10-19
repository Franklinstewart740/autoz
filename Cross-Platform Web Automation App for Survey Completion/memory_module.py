import time
from collections import deque
from typing import Any, Dict, List, Optional
import logging

class MemoryModule:
    """
    Manages short-term memory for agents, allowing them to retain context
    across multi-page surveys and for coherent responses.
    """

    def __init__(self, agent_id: str, capacity: int = 20, ttl: int = 3600):
        self.agent_id = agent_id
        self.capacity = capacity  # Max number of memory entries
        self.ttl = ttl            # Time-to-live for memory entries in seconds
        self.memory_store = deque(maxlen=capacity)  # Stores (timestamp, key, value)
        self.logger = logging.getLogger(f"MemoryModule-{agent_id}")

    def add_entry(self, key: str, value: Any, tags: Optional[List[str]] = None) -> None:
        """
        Adds a new entry to the agent's memory.
        
        Args:
            key: A unique identifier for the memory entry (e.g., 'current_survey_id', 'last_question_text').
            value: The data to store.
            tags: Optional list of tags for categorization (e.g., ['survey_context', 'question_data']).
        """
        timestamp = time.time()
        entry = {
            "timestamp": timestamp,
            "key": key,
            "value": value,
            "tags": tags if tags is not None else [],
            "expires_at": timestamp + self.ttl
        }
        self.memory_store.append(entry)
        self.logger.debug(f"Agent {self.agent_id} added memory: {key}={value}")

    def get_entry(self, key: str) -> Optional[Any]:
        """
        Retrieves the most recent entry for a given key, if it hasn't expired.
        
        Args:
            key: The key of the memory entry to retrieve.
            
        Returns:
            The value of the memory entry, or None if not found or expired.
        """
        self._clean_expired_entries()
        for entry in reversed(self.memory_store):
            if entry["key"] == key:
                self.logger.debug(f"Agent {self.agent_id} retrieved memory: {key}={entry['value']}")
                return entry["value"]
        self.logger.debug(f"Agent {self.agent_id} memory not found for key: {key}")
        return None

    def get_entries_by_tag(self, tag: str) -> List[Any]:
        """
        Retrieves all non-expired entries associated with a specific tag.
        
        Args:
            tag: The tag to search for.
            
        Returns:
            A list of values for matching memory entries.
        """
        self._clean_expired_entries()
        return [entry["value"] for entry in self.memory_store if tag in entry["tags"]]

    def get_all_entries(self) -> List[Dict[str, Any]]:
        """
        Returns all non-expired memory entries.
        """
        self._clean_expired_entries()
        return list(self.memory_store)

    def clear_entry(self, key: str) -> None:
        """
        Removes all entries associated with a specific key.
        """
        initial_len = len(self.memory_store)
        self.memory_store = deque([entry for entry in self.memory_store if entry["key"] != key], maxlen=self.capacity)
        if len(self.memory_store) < initial_len:
            self.logger.debug(f"Agent {self.agent_id} cleared memory for key: {key}")

    def clear_all(self) -> None:
        """
        Clears all memory entries.
        """
        self.memory_store.clear()
        self.logger.debug(f"Agent {self.agent_id} cleared all memory.")

    def _clean_expired_entries(self) -> None:
        """
        Removes expired memory entries from the store.
        """
        current_time = time.time()
        # Filter out expired entries from the left (oldest)
        while self.memory_store and self.memory_store[0]["expires_at"] < current_time:
            expired_entry = self.memory_store.popleft()
            self.logger.debug(f"Agent {self.agent_id} expired memory: {expired_entry['key']}")

    def get_memory_summary(self) -> Dict[str, Any]:
        """
        Provides a summary of the current memory state.
        """
        self._clean_expired_entries()
        unique_keys = set(entry["key"] for entry in self.memory_store)
        return {
            "agent_id": self.agent_id,
            "current_entries": len(self.memory_store),
            "capacity": self.capacity,
            "unique_keys": list(unique_keys),
            "oldest_entry_age_seconds": (time.time() - self.memory_store[0]["timestamp"]) if self.memory_store else 0
        }

