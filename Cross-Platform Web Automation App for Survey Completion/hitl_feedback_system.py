"""
Advanced Human-in-the-Loop (HITL) Feedback System
Enables users to provide explicit feedback for retraining AI models and updating templates.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class FeedbackType(Enum):
    """Types of feedback that users can provide."""
    RESPONSE_QUALITY = "response_quality"
    RESPONSE_CORRECTION = "response_correction"
    TEMPLATE_UPDATE = "template_update"
    PERSONA_FEEDBACK = "persona_feedback"
    DETECTION_ALERT = "detection_alert"
    QUESTION_CLARIFICATION = "question_clarification"


class FeedbackSeverity(Enum):
    """Severity levels for feedback."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FeedbackEntry:
    """Represents a single piece of user feedback."""
    feedback_id: str
    feedback_type: str
    severity: str
    survey_id: str
    question_id: str
    agent_id: str
    original_response: str
    user_suggestion: Optional[str]
    reasoning: str
    timestamp: float
    processed: bool = False
    impact_score: float = 0.0
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class FeedbackAggregator:
    """
    Aggregates feedback from multiple sources and identifies patterns.
    """

    def __init__(self):
        """Initialize the feedback aggregator."""
        self.logger = logging.getLogger("feedback_aggregator")
        self.feedback_entries: Dict[str, FeedbackEntry] = {}
        self.feedback_patterns: Dict[str, List[FeedbackEntry]] = {}
        self.agent_feedback_stats: Dict[str, Dict[str, Any]] = {}

    def add_feedback(self, feedback: FeedbackEntry) -> str:
        """
        Add a feedback entry to the system.
        
        Args:
            feedback: The feedback entry to add.
            
        Returns:
            The feedback ID.
        """
        self.feedback_entries[feedback.feedback_id] = feedback
        
        # Categorize by feedback type
        if feedback.feedback_type not in self.feedback_patterns:
            self.feedback_patterns[feedback.feedback_type] = []
        self.feedback_patterns[feedback.feedback_type].append(feedback)
        
        # Update agent statistics
        if feedback.agent_id not in self.agent_feedback_stats:
            self.agent_feedback_stats[feedback.agent_id] = {
                "total_feedback": 0,
                "positive_feedback": 0,
                "negative_feedback": 0,
                "avg_impact_score": 0.0
            }
        
        stats = self.agent_feedback_stats[feedback.agent_id]
        stats["total_feedback"] += 1
        
        self.logger.info(f"Added feedback {feedback.feedback_id} for agent {feedback.agent_id}")
        return feedback.feedback_id

    def identify_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Identify patterns in feedback across multiple entries.
        
        Returns:
            Dictionary of identified patterns organized by type.
        """
        patterns = {}
        
        # Identify common response issues
        response_issues = {}
        for feedback in self.feedback_entries.values():
            if feedback.feedback_type == FeedbackType.RESPONSE_QUALITY.value:
                issue_key = f"{feedback.survey_id}|{feedback.question_id}"
                if issue_key not in response_issues:
                    response_issues[issue_key] = []
                response_issues[issue_key].append(feedback)
        
        # Find issues with multiple feedback entries
        patterns["common_response_issues"] = [
            {
                "question": key,
                "feedback_count": len(entries),
                "severity_levels": [e.severity for e in entries],
                "suggestions": [e.user_suggestion for e in entries if e.user_suggestion]
            }
            for key, entries in response_issues.items() if len(entries) > 1
        ]
        
        # Identify agent performance patterns
        agent_patterns = {}
        for agent_id, stats in self.agent_feedback_stats.items():
            if stats["total_feedback"] > 0:
                agent_patterns[agent_id] = {
                    "total_feedback": stats["total_feedback"],
                    "feedback_rate": stats["total_feedback"] / max(1, stats["total_feedback"]),
                    "needs_improvement": stats["total_feedback"] > 5
                }
        
        patterns["agent_performance"] = agent_patterns
        
        self.logger.info(f"Identified {len(patterns['common_response_issues'])} common response issues")
        return patterns

    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get a summary of all feedback."""
        total_feedback = len(self.feedback_entries)
        processed_feedback = sum(1 for f in self.feedback_entries.values() if f.processed)
        
        feedback_by_type = {}
        for ftype, entries in self.feedback_patterns.items():
            feedback_by_type[ftype] = len(entries)
        
        return {
            "total_feedback": total_feedback,
            "processed_feedback": processed_feedback,
            "pending_feedback": total_feedback - processed_feedback,
            "feedback_by_type": feedback_by_type,
            "agent_stats": self.agent_feedback_stats
        }


class ModelRetrainingCoordinator:
    """
    Coordinates the retraining of AI models based on user feedback.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the model retraining coordinator.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config or {}
        self.logger = logging.getLogger("model_retraining")
        self.training_datasets: Dict[str, List[Dict[str, Any]]] = {}
        self.retraining_history: List[Dict[str, Any]] = []

    def prepare_training_data(self, feedback_entries: List[FeedbackEntry]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Prepare training data from feedback entries.
        
        Args:
            feedback_entries: List of feedback entries.
            
        Returns:
            Dictionary of training datasets organized by model type.
        """
        training_data = {
            "response_quality": [],
            "response_generation": [],
            "question_understanding": []
        }
        
        for feedback in feedback_entries:
            if feedback.feedback_type == FeedbackType.RESPONSE_QUALITY.value:
                training_sample = {
                    "question": feedback.question_id,
                    "original_response": feedback.original_response,
                    "corrected_response": feedback.user_suggestion,
                    "reasoning": feedback.reasoning,
                    "timestamp": feedback.timestamp
                }
                training_data["response_quality"].append(training_sample)
                training_data["response_generation"].append(training_sample)
            
            elif feedback.feedback_type == FeedbackType.QUESTION_CLARIFICATION.value:
                training_sample = {
                    "question": feedback.question_id,
                    "clarification": feedback.user_suggestion,
                    "reasoning": feedback.reasoning,
                    "timestamp": feedback.timestamp
                }
                training_data["question_understanding"].append(training_sample)
        
        self.training_datasets = training_data
        self.logger.info(f"Prepared training data: {len(training_data['response_quality'])} response samples, "
                        f"{len(training_data['question_understanding'])} clarification samples")
        return training_data

    def schedule_retraining(self, model_type: str, priority: str = "normal") -> Dict[str, Any]:
        """
        Schedule a model retraining job.
        
        Args:
            model_type: Type of model to retrain (e.g., 'response_generation', 'question_understanding').
            priority: Priority level ('low', 'normal', 'high').
            
        Returns:
            Dictionary with retraining job details.
        """
        job_id = f"retrain_{int(time.time() * 1000)}"
        
        training_job = {
            "job_id": job_id,
            "model_type": model_type,
            "priority": priority,
            "scheduled_at": time.time(),
            "status": "scheduled",
            "training_samples": len(self.training_datasets.get(model_type, [])),
            "estimated_duration": self._estimate_training_duration(model_type)
        }
        
        self.retraining_history.append(training_job)
        self.logger.info(f"Scheduled retraining job {job_id} for {model_type} with priority {priority}")
        return training_job

    def _estimate_training_duration(self, model_type: str) -> float:
        """Estimate the duration of model retraining."""
        sample_count = len(self.training_datasets.get(model_type, []))
        # Rough estimate: 0.1 seconds per sample
        return max(60, sample_count * 0.1)  # Minimum 60 seconds

    def get_retraining_status(self) -> Dict[str, Any]:
        """Get the status of retraining operations."""
        return {
            "total_jobs": len(self.retraining_history),
            "scheduled_jobs": sum(1 for j in self.retraining_history if j["status"] == "scheduled"),
            "completed_jobs": sum(1 for j in self.retraining_history if j["status"] == "completed"),
            "training_datasets": {k: len(v) for k, v in self.training_datasets.items()},
            "recent_jobs": self.retraining_history[-5:] if self.retraining_history else []
        }


class TemplateUpdateManager:
    """
    Manages updates to semantic anchoring templates based on user feedback.
    """

    def __init__(self):
        """Initialize the template update manager."""
        self.logger = logging.getLogger("template_update_manager")
        self.template_updates: List[Dict[str, Any]] = []
        self.update_history: List[Dict[str, Any]] = []

    def propose_template_update(self, feedback: FeedbackEntry, template_id: str) -> Dict[str, Any]:
        """
        Propose an update to a semantic anchoring template based on feedback.
        
        Args:
            feedback: The feedback entry.
            template_id: ID of the template to update.
            
        Returns:
            Dictionary with proposed template update.
        """
        update_proposal = {
            "proposal_id": f"update_{int(time.time() * 1000)}",
            "template_id": template_id,
            "feedback_id": feedback.feedback_id,
            "original_response": feedback.original_response,
            "suggested_response": feedback.user_suggestion,
            "reasoning": feedback.reasoning,
            "timestamp": time.time(),
            "status": "pending_review",
            "confidence_score": self._calculate_update_confidence(feedback)
        }
        
        self.template_updates.append(update_proposal)
        self.logger.info(f"Proposed template update {update_proposal['proposal_id']} for template {template_id}")
        return update_proposal

    def approve_template_update(self, proposal_id: str) -> bool:
        """
        Approve a proposed template update.
        
        Args:
            proposal_id: ID of the proposal to approve.
            
        Returns:
            True if approved, False otherwise.
        """
        for update in self.template_updates:
            if update["proposal_id"] == proposal_id:
                update["status"] = "approved"
                update["approved_at"] = time.time()
                
                # Record in history
                self.update_history.append(update)
                self.logger.info(f"Approved template update {proposal_id}")
                return True
        
        return False

    def get_pending_updates(self) -> List[Dict[str, Any]]:
        """Get all pending template update proposals."""
        return [u for u in self.template_updates if u["status"] == "pending_review"]

    def _calculate_update_confidence(self, feedback: FeedbackEntry) -> float:
        """Calculate confidence score for a template update."""
        confidence = 0.5
        
        # Higher confidence if feedback is from multiple sources
        if feedback.severity == FeedbackSeverity.HIGH.value:
            confidence += 0.2
        
        if feedback.user_suggestion:
            confidence += 0.2
        
        if feedback.reasoning:
            confidence += 0.1
        
        return min(1.0, confidence)


class HITLFeedbackSystem:
    """
    Main system coordinating all human-in-the-loop feedback operations.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the HITL Feedback System.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config or {}
        self.logger = logging.getLogger("hitl_system")
        
        self.feedback_aggregator = FeedbackAggregator()
        self.retraining_coordinator = ModelRetrainingCoordinator(config)
        self.template_update_manager = TemplateUpdateManager()

    def submit_feedback(self, feedback_type: str, severity: str, survey_id: str,
                       question_id: str, agent_id: str, original_response: str,
                       user_suggestion: Optional[str] = None, reasoning: str = "") -> str:
        """
        Submit feedback through the HITL system.
        
        Args:
            feedback_type: Type of feedback.
            severity: Severity level.
            survey_id: ID of the survey.
            question_id: ID of the question.
            agent_id: ID of the agent that generated the response.
            original_response: The original response from the agent.
            user_suggestion: User's suggested correction or alternative.
            reasoning: User's reasoning for the feedback.
            
        Returns:
            The feedback ID.
        """
        feedback_id = f"feedback_{int(time.time() * 1000)}"
        
        feedback = FeedbackEntry(
            feedback_id=feedback_id,
            feedback_type=feedback_type,
            severity=severity,
            survey_id=survey_id,
            question_id=question_id,
            agent_id=agent_id,
            original_response=original_response,
            user_suggestion=user_suggestion,
            reasoning=reasoning,
            timestamp=time.time()
        )
        
        self.feedback_aggregator.add_feedback(feedback)
        self.logger.info(f"Submitted feedback {feedback_id} of type {feedback_type}")
        return feedback_id

    def process_feedback_batch(self) -> Dict[str, Any]:
        """
        Process a batch of feedback entries for model retraining and template updates.
        
        Returns:
            Dictionary with processing results.
        """
        # Identify patterns
        patterns = self.feedback_aggregator.identify_patterns()
        
        # Prepare training data
        feedback_list = list(self.feedback_aggregator.feedback_entries.values())
        training_data = self.retraining_coordinator.prepare_training_data(feedback_list)
        
        # Schedule retraining for models with sufficient data
        retraining_jobs = []
        for model_type, samples in training_data.items():
            if len(samples) > 5:  # Minimum samples for retraining
                job = self.retraining_coordinator.schedule_retraining(model_type, priority="normal")
                retraining_jobs.append(job)
        
        # Mark feedback as processed
        for feedback in feedback_list:
            feedback.processed = True
        
        self.logger.info(f"Processed {len(feedback_list)} feedback entries, scheduled {len(retraining_jobs)} retraining jobs")
        
        return {
            "feedback_processed": len(feedback_list),
            "patterns_identified": len(patterns),
            "retraining_jobs_scheduled": len(retraining_jobs),
            "retraining_jobs": retraining_jobs
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Get the overall status of the HITL system."""
        return {
            "feedback_summary": self.feedback_aggregator.get_feedback_summary(),
            "retraining_status": self.retraining_coordinator.get_retraining_status(),
            "pending_template_updates": len(self.template_update_manager.get_pending_updates())
        }

