import logging
from typing import Any, Dict, List, Tuple


class VotingMechanism:
    """
    Implements various voting mechanisms for multi-agent consensus.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger("voting_mechanism")

    def majority_vote(self, proposals: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Selects the proposal with the most votes.
        
        Args:
            proposals: A list of dictionaries, where each dictionary represents a proposal
                       and must contain at least a 'value' key and optionally 'agent_id' and 'confidence'.
                       Example: [{'agent_id': 'R1', 'value': 'Option A', 'confidence': 0.8}]
        
        Returns:
            The winning proposal (dictionary) or None if no consensus.
        """
        if not proposals:
            return None

        vote_counts = {}
        for proposal in proposals:
            value = proposal.get('value')
            if value is not None:
                vote_counts[value] = vote_counts.get(value, 0) + 1
        
        if not vote_counts:
            return None

        max_votes = 0
        winning_value = None
        for value, count in vote_counts.items():
            if count > max_votes:
                max_votes = count
                winning_value = value
            elif count == max_votes: # Handle ties
                # For simplicity, if tied, pick the first one encountered or handle based on confidence
                pass # Could add more complex tie-breaking logic here

        # Find the original proposal that corresponds to the winning value
        for proposal in proposals:
            if proposal.get('value') == winning_value:
                self.logger.info(f"Majority vote: '{winning_value}' with {max_votes} votes.")
                return proposal
        
        return None

    def weighted_vote(self, proposals: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Selects the proposal with the highest weighted score, typically based on confidence.
        
        Args:
            proposals: A list of dictionaries, each containing 'value', 'agent_id', and 'confidence'.
                       Example: [{'agent_id': 'R1', 'value': 'Option A', 'confidence': 0.8}]
        
        Returns:
            The winning proposal (dictionary) or None if no proposals.
        """
        if not proposals:
            return None

        weighted_scores = {}
        for proposal in proposals:
            value = proposal.get('value')
            confidence = proposal.get('confidence', 0.5) # Default confidence if not provided
            if value is not None:
                weighted_scores[value] = weighted_scores.get(value, 0.0) + confidence
        
        if not weighted_scores:
            return None

        max_score = -1.0
        winning_value = None
        for value, score in weighted_scores.items():
            if score > max_score:
                max_score = score
                winning_value = value

        # Find the original proposal that corresponds to the winning value
        for proposal in proposals:
            if proposal.get('value') == winning_value:
                self.logger.info(f"Weighted vote: '{winning_value}' with total score {max_score:.2f}.")
                return proposal
        
        return None

    def confidence_threshold_filter(self, proposals: List[Dict[str, Any]], min_confidence: float = 0.7) -> List[Dict[str, Any]]:
        """
        Filters proposals, keeping only those above a certain confidence threshold.
        
        Args:
            proposals: List of proposals with 'confidence' scores.
            min_confidence: The minimum confidence score required.
            
        Returns:
            A list of proposals that meet the confidence threshold.
        """
        filtered_proposals = [p for p in proposals if p.get('confidence', 0.0) >= min_confidence]
        self.logger.debug(f"Filtered {len(proposals)} proposals to {len(filtered_proposals)} above {min_confidence} confidence.")
        return filtered_proposals

    def get_consensus(self, proposals: List[Dict[str, Any]], method: str = 'majority') -> Optional[Dict[str, Any]]:
        """
        Applies a specified consensus method to a list of proposals.
        
        Args:
            proposals: List of proposals.
            method: The consensus method to use ('majority', 'weighted').
            
        Returns:
            The agreed-upon proposal or None if no consensus.
        """
        if method == 'majority':
            return self.majority_vote(proposals)
        elif method == 'weighted':
            return self.weighted_vote(proposals)
        else:
            self.logger.warning(f"Unknown consensus method: {method}. Defaulting to majority vote.")
            return self.majority_vote(proposals)

    def resolve_disagreement(self, proposals: List[Dict[str, Any]], disagreement_threshold: float = 0.5) -> Optional[Dict[str, Any]]:
        """
        Attempts to resolve disagreement among proposals. If no strong consensus,
        it might return None or a 'needs_human_review' flag.
        
        Args:
            proposals: List of proposals.
            disagreement_threshold: If the top two proposals are too close, it's a disagreement.
            
        Returns:
            The chosen proposal or a signal for human review.
        """
        if not proposals:
            return None

        # Use weighted vote as a primary indicator
        weighted_scores = {}
        for proposal in proposals:
            value = proposal.get('value')
            confidence = proposal.get('confidence', 0.5)
            if value is not None:
                weighted_scores[value] = weighted_scores.get(value, 0.0) + confidence
        
        if not weighted_scores:
            return None

        sorted_scores = sorted(weighted_scores.items(), key=lambda item: item[1], reverse=True)

        if len(sorted_scores) < 2:
            # Only one unique proposal, or no proposals
            return proposals[0] if proposals else None

        top_score = sorted_scores[0][1]
        second_score = sorted_scores[1][1]

        if (top_score - second_score) / top_score < disagreement_threshold: # Check for close scores
            self.logger.warning("High disagreement detected among proposals. Suggesting human review.")
            return {'value': 'HUMAN_REVIEW_REQUIRED', 'reason': 'High disagreement', 'proposals': proposals}
        else:
            # Return the top proposal
            winning_value = sorted_scores[0][0]
            for proposal in proposals:
                if proposal.get('value') == winning_value:
                    return proposal
        return None

