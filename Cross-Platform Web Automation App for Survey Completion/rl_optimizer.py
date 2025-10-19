"""
Reinforcement Learning Optimizer
Implements RL-based optimization for response strategies and anti-detection techniques.
"""

import json
import logging
import random
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class RLState:
    """Represents a state in the RL environment."""
    survey_type: str
    question_type: str
    persona_type: str
    difficulty_level: str  # easy, medium, hard
    previous_success: bool


@dataclass
class RLAction:
    """Represents an action in the RL environment."""
    response_strategy: str  # e.g., "aggressive", "conservative", "neutral"
    anti_detection_level: str  # e.g., "minimal", "moderate", "maximum"
    retry_strategy: str  # e.g., "immediate", "delayed", "adaptive"
    llm_model: str  # e.g., "gpt-3.5", "groq", "ollama"


@dataclass
class RLReward:
    """Represents the reward signal from an action."""
    completion_success: bool
    response_quality: float  # 0.0 to 1.0
    detection_risk: float  # 0.0 to 1.0 (lower is better)
    time_efficiency: float  # 0.0 to 1.0
    user_satisfaction: float  # 0.0 to 1.0
    total_reward: float = 0.0

    def calculate_total(self, weights: Dict[str, float] = None) -> float:
        """Calculate weighted total reward."""
        if weights is None:
            weights = {
                "completion": 0.4,
                "quality": 0.2,
                "detection": 0.2,
                "efficiency": 0.1,
                "satisfaction": 0.1
            }
        
        self.total_reward = (
            weights["completion"] * (1.0 if self.completion_success else 0.0) +
            weights["quality"] * self.response_quality +
            weights["detection"] * (1.0 - self.detection_risk) +
            weights["efficiency"] * self.time_efficiency +
            weights["satisfaction"] * self.user_satisfaction
        )
        return self.total_reward


class QLearningAgent:
    """
    Q-Learning based agent for optimizing survey response strategies.
    """

    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.95, 
                 exploration_rate: float = 0.1):
        """
        Initialize the Q-Learning agent.
        
        Args:
            learning_rate: Learning rate for Q-value updates (0.0 to 1.0).
            discount_factor: Discount factor for future rewards (0.0 to 1.0).
            exploration_rate: Epsilon for epsilon-greedy exploration (0.0 to 1.0).
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.logger = logging.getLogger("rl_optimizer")
        
        # Q-table: state -> action -> Q-value
        self.q_table: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.state_action_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.episode_history: List[Dict[str, Any]] = []

    def select_action(self, state: RLState, available_actions: List[RLAction]) -> RLAction:
        """
        Select an action using epsilon-greedy strategy.
        
        Args:
            state: The current state.
            available_actions: List of available actions.
            
        Returns:
            The selected RLAction.
        """
        state_key = self._state_to_key(state)
        
        # Epsilon-greedy exploration
        if random.random() < self.exploration_rate:
            # Explore: select random action
            selected_action = random.choice(available_actions)
            self.logger.debug(f"Exploration: selected random action {selected_action.response_strategy}")
        else:
            # Exploit: select best action based on Q-values
            best_q_value = -float('inf')
            best_action = available_actions[0]
            
            for action in available_actions:
                action_key = self._action_to_key(action)
                q_value = self.q_table[state_key].get(action_key, 0.0)
                
                if q_value > best_q_value:
                    best_q_value = q_value
                    best_action = action
            
            selected_action = best_action
            self.logger.debug(f"Exploitation: selected action {selected_action.response_strategy} with Q-value {best_q_value:.4f}")
        
        return selected_action

    def update_q_value(self, state: RLState, action: RLAction, reward: RLReward, 
                       next_state: RLState, next_actions: List[RLAction]) -> float:
        """
        Update Q-value using Q-learning formula.
        
        Args:
            state: The current state.
            action: The action taken.
            reward: The reward received.
            next_state: The resulting next state.
            next_actions: Available actions in the next state.
            
        Returns:
            The updated Q-value.
        """
        state_key = self._state_to_key(state)
        action_key = self._action_to_key(action)
        next_state_key = self._state_to_key(next_state)
        
        # Current Q-value
        current_q = self.q_table[state_key][action_key]
        
        # Max Q-value for next state
        max_next_q = 0.0
        if next_actions:
            max_next_q = max(
                self.q_table[next_state_key].get(self._action_to_key(a), 0.0)
                for a in next_actions
            )
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward.total_reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state_key][action_key] = new_q
        self.state_action_counts[state_key][action_key] += 1
        
        self.logger.debug(f"Updated Q-value for {state_key}|{action_key}: {current_q:.4f} -> {new_q:.4f}")
        return new_q

    def decay_exploration(self, decay_rate: float = 0.995) -> None:
        """
        Decay the exploration rate over time.
        
        Args:
            decay_rate: The rate at which to decay exploration (0.0 to 1.0).
        """
        self.exploration_rate *= decay_rate
        self.exploration_rate = max(0.01, self.exploration_rate)  # Minimum exploration rate
        self.logger.info(f"Exploration rate decayed to {self.exploration_rate:.4f}")

    def _state_to_key(self, state: RLState) -> str:
        """Convert state to a hashable key."""
        return f"{state.survey_type}|{state.question_type}|{state.persona_type}|{state.difficulty_level}"

    def _action_to_key(self, action: RLAction) -> str:
        """Convert action to a hashable key."""
        return f"{action.response_strategy}|{action.anti_detection_level}|{action.retry_strategy}|{action.llm_model}"

    def get_q_table_summary(self) -> Dict[str, Any]:
        """Get a summary of the Q-table."""
        total_states = len(self.q_table)
        total_state_actions = sum(len(actions) for actions in self.q_table.values())
        avg_q_value = sum(
            sum(q_vals.values()) for q_vals in self.q_table.values()
        ) / total_state_actions if total_state_actions > 0 else 0.0
        
        return {
            "total_states": total_states,
            "total_state_actions": total_state_actions,
            "average_q_value": avg_q_value,
            "exploration_rate": self.exploration_rate
        }


class PolicyGradientOptimizer:
    """
    Policy Gradient based optimizer for anti-detection strategy refinement.
    """

    def __init__(self, learning_rate: float = 0.001):
        """
        Initialize the Policy Gradient optimizer.
        
        Args:
            learning_rate: Learning rate for policy updates.
        """
        self.learning_rate = learning_rate
        self.logger = logging.getLogger("policy_gradient")
        
        # Policy parameters for anti-detection strategies
        self.policy_params: Dict[str, float] = {
            "proxy_rotation_frequency": 0.5,  # 0.0 to 1.0
            "fingerprint_randomization": 0.5,
            "behavioral_delay_variance": 0.5,
            "header_obfuscation_level": 0.5,
            "request_throttling": 0.5
        }
        
        self.policy_history: List[Dict[str, Any]] = []

    def update_policy(self, detection_signals: Dict[str, float], rewards: List[float]) -> None:
        """
        Update policy parameters based on detection signals and rewards.
        
        Args:
            detection_signals: Dictionary of detection risk signals.
            rewards: List of rewards from recent episodes.
        """
        # Calculate average reward
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        
        # Calculate policy gradient for each parameter
        for param_name, current_value in self.policy_params.items():
            # Detect which signal corresponds to this parameter
            signal_key = param_name.replace("_", " ").lower()
            detection_risk = detection_signals.get(signal_key, 0.0)
            
            # If detection risk is high, increase the parameter (more evasion)
            # If detection risk is low and reward is high, decrease slightly (maintain balance)
            if detection_risk > 0.5:
                gradient = self.learning_rate * (detection_risk - 0.5)
                self.policy_params[param_name] = min(1.0, current_value + gradient)
                self.logger.info(f"Increased {param_name} to {self.policy_params[param_name]:.3f} (detection risk: {detection_risk:.3f})")
            elif avg_reward > 0.7:
                gradient = self.learning_rate * (0.5 - detection_risk) * 0.1
                self.policy_params[param_name] = max(0.0, current_value - gradient)
                self.logger.info(f"Decreased {param_name} to {self.policy_params[param_name]:.3f} (high reward)")
        
        # Record update
        self.policy_history.append({
            "timestamp": time.time(),
            "policy_params": self.policy_params.copy(),
            "avg_reward": avg_reward,
            "detection_signals": detection_signals
        })

    def get_recommended_strategy(self) -> Dict[str, Any]:
        """Get the current recommended anti-detection strategy."""
        return {
            "proxy_rotation": "aggressive" if self.policy_params["proxy_rotation_frequency"] > 0.7 else "moderate" if self.policy_params["proxy_rotation_frequency"] > 0.3 else "minimal",
            "fingerprint_randomization": "aggressive" if self.policy_params["fingerprint_randomization"] > 0.7 else "moderate" if self.policy_params["fingerprint_randomization"] > 0.3 else "minimal",
            "behavioral_delay": "high" if self.policy_params["behavioral_delay_variance"] > 0.7 else "medium" if self.policy_params["behavioral_delay_variance"] > 0.3 else "low",
            "header_obfuscation": "aggressive" if self.policy_params["header_obfuscation_level"] > 0.7 else "moderate" if self.policy_params["header_obfuscation_level"] > 0.3 else "minimal",
            "request_throttling": "strict" if self.policy_params["request_throttling"] > 0.7 else "moderate" if self.policy_params["request_throttling"] > 0.3 else "relaxed"
        }


class RLOptimizationEngine:
    """
    Main engine coordinating Q-Learning and Policy Gradient optimization.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the RL Optimization Engine.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config or {}
        self.logger = logging.getLogger("rl_engine")
        
        self.q_agent = QLearningAgent(
            learning_rate=self.config.get("learning_rate", 0.1),
            discount_factor=self.config.get("discount_factor", 0.95),
            exploration_rate=self.config.get("exploration_rate", 0.1)
        )
        
        self.policy_optimizer = PolicyGradientOptimizer(
            learning_rate=self.config.get("policy_learning_rate", 0.001)
        )
        
        self.episode_count = 0
        self.total_rewards = []

    def process_episode(self, state: RLState, action: RLAction, reward: RLReward,
                       next_state: RLState, next_actions: List[RLAction],
                       detection_signals: Dict[str, float]) -> None:
        """
        Process a complete episode and update both Q-learning and policy.
        
        Args:
            state: Initial state.
            action: Action taken.
            reward: Reward received.
            next_state: Resulting state.
            next_actions: Available actions in next state.
            detection_signals: Detection risk signals.
        """
        # Update Q-value
        self.q_agent.update_q_value(state, action, reward, next_state, next_actions)
        
        # Accumulate rewards for policy update
        self.total_rewards.append(reward.total_reward)
        
        # Periodically update policy and decay exploration
        self.episode_count += 1
        if self.episode_count % 10 == 0:
            # Update policy based on recent rewards and detection signals
            recent_rewards = self.total_rewards[-10:]
            self.policy_optimizer.update_policy(detection_signals, recent_rewards)
            
            # Decay exploration rate
            self.q_agent.decay_exploration()
            
            self.logger.info(f"Episode {self.episode_count}: Avg reward = {sum(recent_rewards)/len(recent_rewards):.4f}")

    def get_optimization_status(self) -> Dict[str, Any]:
        """Get the current optimization status."""
        return {
            "episode_count": self.episode_count,
            "q_table_summary": self.q_agent.get_q_table_summary(),
            "recommended_strategy": self.policy_optimizer.get_recommended_strategy(),
            "average_recent_reward": sum(self.total_rewards[-10:]) / len(self.total_rewards[-10:]) if len(self.total_rewards) >= 10 else 0.0
        }

