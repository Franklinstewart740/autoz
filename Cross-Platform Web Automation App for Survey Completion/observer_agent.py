"""
Observer Agent
Passive agent responsible for monitoring and logging the decisions and interactions of active agents.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from .base_agent import BaseAgent, AgentTask, AgentMessage


class ObserverAgent(BaseAgent):
    """
    Observer Agent passively monitors and logs agent activities.
    Specialized in:
    - Logging agent decisions and interactions
    - Benchmarking agent performance
    - Generating data for debugging and training
    """

    def __init__(self, agent_id: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(agent_id, "observer", config)
        
        self.log_file = config.get("log_file", "agent_swarm.log")
        self.log_handler = None
        self.event_log: List[Dict[str, Any]] = []

    async def initialize(self) -> bool:
        """Initialize the Observer Agent with logging setup."""
        try:
            self.logger.info("Initializing Observer Agent")
            
            # Set up dedicated log file for swarm events
            self.log_handler = logging.FileHandler(self.log_file)
            self.log_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            
            swarm_logger = logging.getLogger("swarm_events")
            swarm_logger.addHandler(self.log_handler)
            swarm_logger.setLevel(logging.INFO)
            
            self.logger.info(f"Observer Agent logging to {self.log_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Observer Agent: {e}", exc_info=True)
            return False

    async def cleanup(self) -> None:
        """Clean up logging resources."""
        if self.log_handler:
            self.log_handler.close()
        self.logger.info("Observer Agent cleanup completed")

    async def process_task(self, task: AgentTask) -> Dict[str, Any]:
        """
        Observer agent is passive and does not process tasks directly.
        """
        self.logger.warning("Observer agent does not process tasks.")
        return {"success": False, "error": "Observer agent is passive"}

    async def handle_custom_message(self, message: AgentMessage) -> None:
        """
        Handle incoming messages by logging them.
        This is the primary function of the Observer Agent.
        """
        event = {
            "timestamp": time.time(),
            "message_id": message.id,
            "sender": message.sender,
            "recipient": message.recipient,
            "message_type": message.message_type,
            "payload": message.payload
        }
        
        self.event_log.append(event)
        
        # Log to dedicated swarm log file
        swarm_logger = logging.getLogger("swarm_events")
        swarm_logger.info(f"EVENT: {event}")
        
        self.logger.debug(f"Observed and logged message {message.id} from {message.sender}")

    def get_event_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get the most recent events from the log.
        
        Args:
            limit: Maximum number of events to return
            
        Returns:
            List of event dictionaries
        """
        return self.event_log[-limit:]

    def get_agent_performance_summary(self) -> Dict[str, Any]:
        """
        Analyze the event log to generate a performance summary for all agents.
        """
        summary = {}
        
        for event in self.event_log:
            if event["message_type"] == "status_response":
                agent_id = event["sender"]
                payload = event["payload"]
                
                if agent_id not in summary:
                    summary[agent_id] = {
                        "tasks_completed": 0,
                        "tasks_failed": 0,
                        "total_processing_time": 0.0,
                        "average_processing_time": 0.0
                    }
                
                # This is a simplified summary. A more robust implementation
                # would track tasks from assignment to completion.
                if "metrics" in payload:
                    summary[agent_id] = payload["metrics"]
                    
        return summary

    def analyze_events(self) -> Dict[str, Any]:
        """
        Analyze the event log for behavioral patterns, common errors, and task flow.
        
        Returns:
            Dictionary with analysis results
        """
        analysis = {
            "total_events": len(self.event_log),
            "agent_interactions": {},
            "task_flow_summary": {},
            "error_patterns": {},
            "agent_decision_frequency": {}
        }

        for event in self.event_log:
            sender = event["sender"]
            recipient = event["recipient"]
            message_type = event["message_type"]
            payload = event["payload"]

            # Agent interactions
            if sender not in analysis["agent_interactions"]:
                analysis["agent_interactions"][sender] = {}
            analysis["agent_interactions"][sender][recipient] = \
                analysis["agent_interactions"][sender].get(recipient, 0) + 1

            # Task flow summary (simplified)
            if message_type == "task_assigned":
                task_type = payload.get("task_type", "unknown")
                analysis["task_flow_summary"][task_type] = \
                    analysis["task_flow_summary"].get(task_type, 0) + 1
            elif message_type == "task_result" and not payload.get("success", True):
                error_type = payload.get("error_type", "generic_error")
                analysis["error_patterns"][error_type] = \
                    analysis["error_patterns"].get(error_type, 0) + 1
            
            # Agent decision frequency (if payload contains decision info)
            if "decision" in payload:
                decision = payload["decision"]
                if sender not in analysis["agent_decision_frequency"]:
                    analysis["agent_decision_frequency"][sender] = {}
                analysis["agent_decision_frequency"][sender][decision] = \
                    analysis["agent_decision_frequency"][sender].get(decision, 0) + 1

        return analysis

