import asyncio
import logging
from typing import Any, Dict, List, Optional

from .base_agent import BaseAgent, AgentTask, AgentMessage
from .navigator_agent import NavigatorAgent
from .extractor_agent import ExtractorAgent
from .responder_agent import ResponderAgent
from .validator_agent import ValidatorAgent
from .observer_agent import ObserverAgent
from .voting_mechanism import VotingMechanism


class AgentOrchestrator:
    """
    Manages the lifecycle and coordination of all specialized agents.
    - Initializes and manages agent instances
    - Dispatches tasks to appropriate agents
    - Facilitates inter-agent communication
    - Implements high-level survey automation workflows
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("orchestrator")
        
        self.agents: Dict[str, BaseAgent] = {}
        self.task_queue = asyncio.Queue()
        self.message_bus = asyncio.Queue()
        self.observer: Optional[ObserverAgent] = None
        self.voting_mechanism = VotingMechanism(config.get("voting_mechanism", {}))
        
        self.is_running = False

    async def start(self) -> None:
        """Initialize and start all agents and the main orchestration loop."""
        if self.is_running:
            self.logger.warning("Orchestrator is already running.")
            return

        self.logger.info("Starting Agent Orchestrator")
        
        # Initialize agents
        await self._initialize_agents()
        
        # Start agent processing loops
        for agent in self.agents.values():
            asyncio.create_task(agent.start())
        
        # Start orchestrator loops
        self.is_running = True
        asyncio.create_task(self._task_dispatch_loop())
        asyncio.create_task(self._message_routing_loop())
        
        self.logger.info("Agent Orchestrator started successfully")

    async def stop(self) -> None:
        """Stop all agents and the orchestrator."""
        if not self.is_running:
            return
            
        self.logger.info("Stopping Agent Orchestrator")
        self.is_running = False
        
        cleanup_tasks = [agent.cleanup() for agent in self.agents.values()]
        await asyncio.gather(*cleanup_tasks)
            
        self.logger.info("Agent Orchestrator stopped")

    async def _initialize_agents(self) -> None:
        """Create and initialize all agent instances."""
        agent_configs = self.config.get("agents", {})
        
        # Observer Agent (special case)
        self.observer = ObserverAgent("observer-1", agent_configs.get("observer", {}))
        self.agents["observer-1"] = self.observer
        
        # Active Agents
        self.agents["navigator-1"] = NavigatorAgent("navigator-1", agent_configs.get("navigator", {}))
        self.agents["extractor-1"] = ExtractorAgent("extractor-1", agent_configs.get("extractor", {}))
        self.agents["responder-1"] = ResponderAgent("responder-1", agent_configs.get("responder", {}))
        self.agents["validator-1"] = ValidatorAgent("validator-1", agent_configs.get("validator", {}))
        
        # Initialize all agents
        init_tasks = [agent.initialize() for agent in self.agents.values()]
        results = await asyncio.gather(*init_tasks)
        
        for i, (agent_id, success) in enumerate(zip(self.agents.keys(), results)):
            if not success:
                self.logger.error(f"Failed to initialize agent: {agent_id}")
                # Optionally, remove the failed agent or mark it as unhealthy
                # del self.agents[agent_id]

        self.logger.info(f"Initialized {len(self.agents)} agents")

    async def submit_task(self, task: AgentTask) -> None:
        """Submit a new task to the orchestrator."""
        await self.task_queue.put(task)
        self.logger.info(f"Submitted task {task.id} to the queue.")

    async def _task_dispatch_loop(self) -> None:
        """Continuously dispatch tasks from the queue to appropriate agents."""
        while self.is_running:
            try:
                task = await self.task_queue.get()
                self.logger.debug(f"Dispatching task {task.id} of type {task.task_type}")
                
                # Simple routing based on task type (can be more sophisticated)
                if task.task_type.startswith("navigate") or task.task_type.startswith("login"):
                    await self.agents["navigator-1"].assign_task(task)
                elif task.task_type.startswith("extract"):
                    await self.agents["extractor-1"].assign_task(task)
                elif task.task_type.startswith("generate"):
                    await self.agents["responder-1"].assign_task(task)
                elif task.task_type.startswith("validate"):
                    await self.agents["validator-1"].assign_task(task)
                else:
                    self.logger.error(f"No agent found to handle task type: {task.task_type}")
                    task.status = "failed"
                    task.error = "No agent available"

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in task dispatch loop: {e}", exc_info=True)

    async def _message_routing_loop(self) -> None:
        """Route messages between agents."""
        # Collect all outbound message queues from agents
        outbound_queues = [agent.outbound_messages for agent in self.agents.values()]
        
        while self.is_running:
            try:
                # Wait for a message from any agent
                # This requires a more complex mechanism to wait on multiple queues.
                # For simplicity, we'll iterate and check, but a better solution
                # would use asyncio.wait or a dedicated message bus library.
                for queue in outbound_queues:
                    if not queue.empty():
                        message = await queue.get()
                        await self._route_message(message)
                
                await asyncio.sleep(0.1) # Prevent busy-waiting

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in message routing loop: {e}", exc_info=True)

    async def _route_message(self, message: AgentMessage) -> None:
        """
        Route a single message to its recipient.
        Also sends a copy to the Observer Agent.
        """
        # Log the message with the Observer
        if self.observer:
            await self.observer.receive_message(message)
            
        recipient_id = message.recipient
        
        if recipient_id == "orchestrator":
            # Handle messages intended for the orchestrator
            await self._handle_orchestrator_message(message)
            
        elif recipient_id in self.agents:
            await self.agents[recipient_id].receive_message(message)
            self.logger.debug(f"Routed message {message.id} to {recipient_id}")
            
        else:
            self.logger.warning(f"Recipient {recipient_id} not found for message {message.id}")

    async def _handle_orchestrator_message(self, message: AgentMessage) -> None:
        """
        Handle messages specifically addressed to the orchestrator.
        """
        self.logger.debug(f"Orchestrator received message: {message.message_type}")
        
        if message.message_type == "task_result":
            # A task has been completed, potentially trigger next step in workflow
            self.logger.info(f"Task {message.payload.get("task_id")} completed by {message.sender}")
            # High-level workflow logic would go here
            
        elif message.message_type == "request_help":
            # An agent needs help, escalate or re-assign
            self.logger.warning(f"Agent {message.sender} requested help: {message.payload.get("reason")}")
            # Escalation logic would go here

    # High-level survey automation workflow
    async def run_survey_workflow(self, platform: str, credentials: Dict[str, str], survey_id: Optional[str] = None) -> None:
        """
        Example of a high-level workflow for completing a survey.
        """
        self.logger.info(f"Starting survey workflow for platform: {platform}")
        
        # 1. Login to platform
        login_task = AgentTask(
            id="login-1", task_type="login_to_platform", 
            payload={"platform": platform, "credentials": credentials}, priority=10
        )
        await self.submit_task(login_task)
        
        # This is a simplified example. A real implementation would wait for task completion
        # and handle results before proceeding to the next step.
        # This requires a more complex state machine or workflow engine.
        
        self.logger.info("Workflow steps submitted. Monitor agent logs for progress.")

    def get_all_agent_statuses(self) -> Dict[str, Any]:
        """Get the status of all managed agents."""
        statuses = {}
        for agent_id, agent in self.agents.items():
            statuses[agent_id] = agent.get_status()
        return statuses

    async def coordinate_responses(self, proposals: List[Dict[str, Any]], method: str = 'weighted') -> Dict[str, Any]:
        """
        Coordinates responses from multiple agents using the configured voting mechanism.
        
        Args:
            proposals: A list of dictionaries, where each dictionary represents a proposal
                       from an agent and must contain at least a 'value' key, 'agent_id', and 'confidence'.
                       Example: [{'agent_id': 'responder-1', 'value': 'Option A', 'confidence': 0.8}]
            method: The consensus method to use ('majority', 'weighted').
            
        Returns:
            A dictionary containing the chosen proposal or a signal for human review.
        """
        self.logger.info(f"Coordinating responses using {method} method for {len(proposals)} proposals.")
        
        # Log proposals for observer
        if self.observer:
            await self.observer.receive_message(AgentMessage(
                id="coord-proposals",
                sender="orchestrator",
                recipient="observer-1",
                message_type="proposals_for_vote",
                payload={"proposals": proposals, "method": method}
            ))

        # Apply voting mechanism
        consensus_result = self.voting_mechanism.get_consensus(proposals, method=method)

        if consensus_result:
            if consensus_result.get('value') == 'HUMAN_REVIEW_REQUIRED':
                self.logger.warning("Consensus not reached, human review required.")
                # Log for observer
                if self.observer:
                    await self.observer.receive_message(AgentMessage(
                        id="coord-human-review",
                        sender="orchestrator",
                        recipient="observer-1",
                        message_type="human_review_needed",
                        payload={"reason": "High disagreement", "proposals": proposals}
                    ))
                return {"status": "human_review_required", "details": consensus_result}
            else:
                self.logger.info(f"Consensus reached: {consensus_result.get('value')}")
                # Log for observer
                if self.observer:
                    await self.observer.receive_message(AgentMessage(
                        id="coord-consensus",
                        sender="orchestrator",
                        recipient="observer-1",
                        message_type="consensus_reached",
                        payload={"result": consensus_result}
                    ))
                return {"status": "success", "result": consensus_result}
        else:
            self.logger.warning("No consensus could be reached from the proposals.")
            return {"status": "no_consensus", "details": "No agreement among agents"}

