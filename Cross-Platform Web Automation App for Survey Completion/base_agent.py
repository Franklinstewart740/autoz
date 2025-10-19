import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from .memory_module import MemoryModule
import uuid


class AgentStatus(Enum):
    """Agent status enumeration."""
    IDLE = "idle"
    WORKING = "working"
    ERROR = "error"
    PAUSED = "paused"
    COMPLETED = "completed"


@dataclass
class AgentMessage:
    """Message structure for inter-agent communication."""
    id: str
    sender: str
    recipient: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: float
    priority: int = 0


@dataclass
class AgentTask:
    """
    Task structure for agent work assignments.
    """
    id: str
    task_type: str
    payload: Dict[str, Any]
    priority: int
    created_at: float
    assigned_to: Optional[str] = None
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class BaseAgent(ABC):
    """
    Base class for all specialized agents in the survey automation system.
    Provides common functionality including communication, task management,
    logging, and error handling.
    """

    def __init__(self, agent_id: str, agent_type: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base agent.
        
        Args:
            agent_id: Unique identifier for this agent instance
            agent_type: Type/role of the agent (e.g., 'navigator', 'extractor')
            config: Optional configuration dictionary
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.config = config or {}
        self.status = AgentStatus.IDLE
        self.logger = logging.getLogger(f"agent.{agent_type}.{agent_id}")
        
        # Communication
        self.message_queue = asyncio.Queue()
        self.outbound_messages = asyncio.Queue()
        
        # Task management
        self.current_task: Optional[AgentTask] = None
        self.task_history: List[AgentTask] = []
        
        # Performance metrics
        self.metrics = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'last_activity': time.time()
        }
        
        # Memory for context retention
        self.memory = MemoryModule(agent_id=self.agent_id, capacity=self.config.get("memory_capacity", 20), ttl=self.config.get("memory_ttl", 3600))
        self.session_context: Dict[str, Any] = {}
        
        self.logger.info(f"Agent {self.agent_id} ({self.agent_type}) initialized")

    @abstractmethod
    async def process_task(self, task: AgentTask) -> Dict[str, Any]:
        """
        Process a task assigned to this agent.
        
        Args:
            task: The task to process
            
        Returns:
            Dictionary containing the task result
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass

    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize agent-specific resources and configurations.
        
        Returns:
            True if initialization successful, False otherwise
        """
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up agent resources."""
        pass

    async def start(self) -> None:
        """Start the agent's main processing loop."""
        self.logger.info(f"Starting agent {self.agent_id}")
        
        if not await self.initialize():
            self.logger.error(f"Failed to initialize agent {self.agent_id}")
            return
        
        self.status = AgentStatus.IDLE
        
        # Start message processing loop
        asyncio.create_task(self._message_processing_loop())
        
        self.logger.info(f"Agent {self.agent_id} started successfully")

    async def stop(self) -> None:
        """
        Stop the agent and clean up resources.
        """
        self.logger.info(f"Stopping agent {self.agent_id}")
        self.status = AgentStatus.IDLE
        await self.cleanup()
        self.logger.info(f"Agent {self.agent_id} stopped")

    async def assign_task(self, task: AgentTask) -> bool:
        """
        Assign a task to this agent.
        
        Args:
            task: The task to assign
            
        Returns:
            True if task was accepted, False otherwise
        """
        if self.status != AgentStatus.IDLE:
            self.logger.warning(f"Agent {self.agent_id} is busy, cannot assign task {task.id}")
            return False
        
        self.current_task = task
        task.assigned_to = self.agent_id
        task.status = "assigned"
        
        self.logger.info(f"Task {task.id} assigned to agent {self.agent_id}")
        
        # Process the task asynchronously
        asyncio.create_task(self._execute_task(task))
        
        return True

    async def _execute_task(self, task: AgentTask) -> None:
        """
        Execute a task and handle the result.
        
        Args:
            task: The task to execute
        """
        start_time = time.time()
        self.status = AgentStatus.WORKING
        task.status = "in_progress"
        
        try:
            self.logger.info(f"Executing task {task.id}")
            result = await self.process_task(task)
            
            task.result = result
            task.status = "completed"
            self.status = AgentStatus.COMPLETED
            
            # Update metrics
            processing_time = time.time() - start_time
            self.metrics['tasks_completed'] += 1
            self.metrics['total_processing_time'] += processing_time
            self.metrics['average_processing_time'] = (
                self.metrics['total_processing_time'] / self.metrics['tasks_completed']
            )
            self.metrics['last_activity'] = time.time()
            
            self.logger.info(f"Task {task.id} completed successfully in {processing_time:.2f}s")
            
        except Exception as e:
            task.error = str(e)
            task.status = "failed"
            self.status = AgentStatus.ERROR
            self.metrics['tasks_failed'] += 1
            
            self.logger.error(f"Task {task.id} failed: {e}", exc_info=True)
        
        finally:
            # Add to history and reset current task
            self.task_history.append(task)
            self.current_task = None
            
            # Reset status to idle after a brief delay
            await asyncio.sleep(0.1)
            if self.status in [AgentStatus.COMPLETED, AgentStatus.ERROR]:
                self.status = AgentStatus.IDLE

    async def send_message(self, recipient: str, message_type: str, payload: Dict[str, Any], priority: int = 0) -> None:
        """
        Send a message to another agent or the orchestrator.
        
        Args:
            recipient: ID of the recipient agent or 'orchestrator'
            message_type: Type of message (e.g., 'task_result', 'request_help')
            payload: Message payload
            priority: Message priority (higher = more urgent)
        """
        message = AgentMessage(
            id=str(uuid.uuid4()),
            sender=self.agent_id,
            recipient=recipient,
            message_type=message_type,
            payload=payload,
            timestamp=time.time(),
            priority=priority
        )
        
        await self.outbound_messages.put(message)
        self.logger.debug(f"Sent message {message.id} to {recipient}")

    async def receive_message(self, message: AgentMessage) -> None:
        """
        Receive a message from another agent or the orchestrator.
        
        Args:
            message: The received message
        """
        await self.message_queue.put(message)
        self.logger.debug(f"Received message {message.id} from {message.sender}")

    async def _message_processing_loop(self) -> None:
        """
        Process incoming messages continuously.
        """
        while True:
            try:
                # Wait for a message with a timeout to allow for graceful shutdown
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                await self._handle_message(message)
            except asyncio.TimeoutError:
                # No message received, continue loop
                continue
            except Exception as e:
                self.logger.error(f"Error processing message: {e}", exc_info=True)

    async def _handle_message(self, message: AgentMessage) -> None:
        """
        Handle an incoming message.
        
        Args:
            message: The message to handle
        """
        self.logger.debug(f"Handling message {message.id} of type {message.message_type}")
        
        if message.message_type == "pause":
            self.status = AgentStatus.PAUSED
            await self.send_message("orchestrator", "status_update", {"status": "paused"})
        
        elif message.message_type == "resume":
            if self.status == AgentStatus.PAUSED:
                self.status = AgentStatus.IDLE
                await self.send_message("orchestrator", "status_update", {"status": "idle"})
        
        elif message.message_type == "status_request":
            await self.send_message(
                message.sender, 
                "status_response", 
                {
                    "status": self.status.value,
                    "current_task": self.current_task.id if self.current_task else None,
                    "metrics": self.metrics
                }
            )
        
        elif message.message_type == "context_update":
            # Update session context with new information
            self.session_context.update(message.payload.get("context", {}))
            self.logger.info("Session context updated")
        
        else:
            # Handle agent-specific messages
            await self.handle_custom_message(message)

    async def handle_custom_message(self, message: AgentMessage) -> None:
        """
        Handle custom message types specific to the agent implementation.
        
        Args:
            message: The message to handle
        """
        # Default implementation does nothing
        # Subclasses can override this method
        pass

    def update_memory(self, key: str, value: Any, tags: Optional[List[str]] = None) -> None:
        """
        Update the agent's short-term memory.
        
        Args:
            key: Memory key
            value: Value to store
            tags: Optional list of tags for categorization.
        """
        self.memory.add_entry(key, value, tags)
        self.logger.debug(f"Updated memory: {key}")

    def get_memory(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a value from the agent's short-term memory.
        
        Args:
            key: Memory key
            default: Default value if key not found
            
        Returns:
            The stored value or default
        """
        return self.memory.get_entry(key) or default

    def clear_memory(self) -> None:
        """
        Clear the agent's short-term memory.
        """
        self.memory.clear_all()
        self.logger.debug("Memory cleared")

    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the agent.
        
        Returns:
            Dictionary containing agent status information
        """
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "status": self.status.value,
            "current_task": self.current_task.id if self.current_task else None,
            "metrics": self.metrics.copy(),
            "memory_size": self.memory.get_memory_summary()["current_entries"],
            "context_size": len(self.session_context)
        }

    def set_confidence_threshold(self, threshold: float) -> None:
        """
        Set the confidence threshold for this agent's decisions.
        
        Args:
            threshold: Confidence threshold (0.0 to 1.0)
        """
        self.config['confidence_threshold'] = max(0.0, min(1.0, threshold))
        self.logger.info(f"Confidence threshold set to {threshold}")

    def get_confidence_threshold(self) -> float:
        """
        Get the current confidence threshold.
        
        Returns:
            Current confidence threshold
        """
        return self.config.get('confidence_threshold', 0.7)

