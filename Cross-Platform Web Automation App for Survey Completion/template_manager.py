"""
Template Manager
Manages known question templates and their associated response strategies.
"""

import logging
import json
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class QuestionTemplate:
    """Represents a known question template with response strategy."""
    id: str
    template_text: str
    category: str
    question_type: str
    response_strategy: Dict[str, Any]
    examples: List[str]
    tags: List[str]
    confidence_threshold: float
    created_at: str
    updated_at: str
    usage_count: int = 0


class TemplateManager:
    """
    Manages a database of known question templates and their response strategies.
    Enables reuse of fine-tuned logic for similar questions.
    """

    def __init__(self, templates_file: Optional[str] = None):
        """
        Initialize the Template Manager.
        
        Args:
            templates_file: Path to JSON file containing templates
        """
        self.logger = logging.getLogger("template_manager")
        self.templates: Dict[str, QuestionTemplate] = {}
        self.templates_file = Path(templates_file) if templates_file else Path("./data/question_templates.json")
        
        # Create data directory if it doesn't exist
        self.templates_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing templates
        self.load_templates()
        
        # Initialize with default templates if none exist
        if not self.templates:
            self._initialize_default_templates()

    def add_template(self, template: QuestionTemplate) -> bool:
        """
        Add a new question template.
        
        Args:
            template: QuestionTemplate object to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            template.updated_at = datetime.now().isoformat()
            self.templates[template.id] = template
            self.logger.info(f"Added template: {template.id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add template {template.id}: {e}", exc_info=True)
            return False

    def get_template(self, template_id: str) -> Optional[QuestionTemplate]:
        """
        Get a template by ID.
        
        Args:
            template_id: Template ID
            
        Returns:
            QuestionTemplate object or None if not found
        """
        return self.templates.get(template_id)

    def get_templates_by_category(self, category: str) -> List[QuestionTemplate]:
        """
        Get all templates in a specific category.
        
        Args:
            category: Template category
            
        Returns:
            List of QuestionTemplate objects
        """
        return [template for template in self.templates.values() if template.category == category]

    def get_templates_by_type(self, question_type: str) -> List[QuestionTemplate]:
        """
        Get all templates of a specific question type.
        
        Args:
            question_type: Question type
            
        Returns:
            List of QuestionTemplate objects
        """
        return [template for template in self.templates.values() if template.question_type == question_type]

    def search_templates(self, query: str, tags: Optional[List[str]] = None) -> List[QuestionTemplate]:
        """
        Search templates by text query and/or tags.
        
        Args:
            query: Text query to search in template text and examples
            tags: List of tags to filter by
            
        Returns:
            List of matching QuestionTemplate objects
        """
        results = []
        query_lower = query.lower()
        
        for template in self.templates.values():
            # Text search
            text_match = (
                query_lower in template.template_text.lower() or
                any(query_lower in example.lower() for example in template.examples)
            )
            
            # Tag search
            tag_match = not tags or any(tag in template.tags for tag in tags)
            
            if text_match and tag_match:
                results.append(template)
        
        return results

    def update_template(self, template_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update an existing template.
        
        Args:
            template_id: Template ID to update
            updates: Dictionary of fields to update
            
        Returns:
            True if successful, False otherwise
        """
        if template_id not in self.templates:
            self.logger.error(f"Template {template_id} not found")
            return False
        
        try:
            template = self.templates[template_id]
            
            # Update fields
            for field, value in updates.items():
                if hasattr(template, field):
                    setattr(template, field, value)
            
            template.updated_at = datetime.now().isoformat()
            
            self.logger.info(f"Updated template: {template_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update template {template_id}: {e}", exc_info=True)
            return False

    def delete_template(self, template_id: str) -> bool:
        """
        Delete a template.
        
        Args:
            template_id: Template ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        if template_id in self.templates:
            del self.templates[template_id]
            self.logger.info(f"Deleted template: {template_id}")
            return True
        else:
            self.logger.warning(f"Template {template_id} not found for deletion")
            return False

    def increment_usage(self, template_id: str) -> None:
        """
        Increment the usage count for a template.
        
        Args:
            template_id: Template ID
        """
        if template_id in self.templates:
            self.templates[template_id].usage_count += 1
            self.templates[template_id].updated_at = datetime.now().isoformat()

    def get_most_used_templates(self, limit: int = 10) -> List[QuestionTemplate]:
        """
        Get the most frequently used templates.
        
        Args:
            limit: Maximum number of templates to return
            
        Returns:
            List of QuestionTemplate objects sorted by usage count
        """
        sorted_templates = sorted(
            self.templates.values(),
            key=lambda t: t.usage_count,
            reverse=True
        )
        return sorted_templates[:limit]

    def get_all_templates(self) -> List[QuestionTemplate]:
        """
        Get all templates.
        
        Returns:
            List of all QuestionTemplate objects
        """
        return list(self.templates.values())

    def get_template_texts(self) -> List[str]:
        """
        Get all template texts for embedding/similarity computation.
        
        Returns:
            List of template text strings
        """
        return [template.template_text for template in self.templates.values()]

    def get_template_ids(self) -> List[str]:
        """
        Get all template IDs.
        
        Returns:
            List of template ID strings
        """
        return list(self.templates.keys())

    def save_templates(self) -> bool:
        """
        Save templates to JSON file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert templates to JSON-serializable format
            templates_data = {
                template_id: asdict(template)
                for template_id, template in self.templates.items()
            }
            
            with open(self.templates_file, 'w') as f:
                json.dump(templates_data, f, indent=2)
            
            self.logger.info(f"Saved {len(self.templates)} templates to {self.templates_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save templates: {e}", exc_info=True)
            return False

    def load_templates(self) -> bool:
        """
        Load templates from JSON file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.templates_file.exists():
                self.logger.info("Templates file does not exist, starting with empty templates")
                return True
            
            with open(self.templates_file, 'r') as f:
                templates_data = json.load(f)
            
            # Convert JSON data to QuestionTemplate objects
            self.templates = {}
            for template_id, template_dict in templates_data.items():
                template = QuestionTemplate(**template_dict)
                self.templates[template_id] = template
            
            self.logger.info(f"Loaded {len(self.templates)} templates from {self.templates_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load templates: {e}", exc_info=True)
            return False

    def export_templates(self, output_file: str, format: str = "json") -> bool:
        """
        Export templates to a file.
        
        Args:
            output_file: Path to output file
            format: Export format ("json", "csv")
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if format == "json":
                templates_data = {
                    template_id: asdict(template)
                    for template_id, template in self.templates.items()
                }
                with open(output_file, 'w') as f:
                    json.dump(templates_data, f, indent=2)
                    
            elif format == "csv":
                import csv
                with open(output_file, 'w', newline='') as f:
                    if not self.templates:
                        return True
                    
                    # Get field names from first template
                    fieldnames = list(asdict(next(iter(self.templates.values()))).keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for template in self.templates.values():
                        row = asdict(template)
                        # Convert lists to strings for CSV
                        row['examples'] = '; '.join(row['examples'])
                        row['tags'] = '; '.join(row['tags'])
                        row['response_strategy'] = json.dumps(row['response_strategy'])
                        writer.writerow(row)
                        
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Exported {len(self.templates)} templates to {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export templates: {e}", exc_info=True)
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the template database.
        
        Returns:
            Dictionary with statistics
        """
        if not self.templates:
            return {"total_templates": 0}
        
        categories = {}
        question_types = {}
        total_usage = 0
        
        for template in self.templates.values():
            # Count by category
            categories[template.category] = categories.get(template.category, 0) + 1
            
            # Count by question type
            question_types[template.question_type] = question_types.get(template.question_type, 0) + 1
            
            # Sum usage
            total_usage += template.usage_count
        
        return {
            "total_templates": len(self.templates),
            "categories": categories,
            "question_types": question_types,
            "total_usage": total_usage,
            "average_usage": total_usage / len(self.templates) if self.templates else 0,
            "most_used_template": max(self.templates.values(), key=lambda t: t.usage_count).id if self.templates else None
        }

    def _initialize_default_templates(self) -> None:
        """Initialize with a set of default question templates."""
        default_templates = [
            QuestionTemplate(
                id="age_question",
                template_text="What is your age?",
                category="demographic",
                question_type="numeric",
                response_strategy={
                    "generation_method": "persona_based",
                    "validation": {"min": 18, "max": 100},
                    "consistency_required": True
                },
                examples=[
                    "How old are you?",
                    "What is your current age?",
                    "Please enter your age"
                ],
                tags=["age", "demographic", "personal"],
                confidence_threshold=0.8,
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat()
            ),
            QuestionTemplate(
                id="satisfaction_likert",
                template_text="How satisfied are you with our service?",
                category="feedback",
                question_type="likert_scale",
                response_strategy={
                    "generation_method": "weighted_random",
                    "scale_bias": "positive",
                    "avoid_extremes": True
                },
                examples=[
                    "Rate your satisfaction with the product",
                    "How would you rate your overall experience?",
                    "Please rate your satisfaction level"
                ],
                tags=["satisfaction", "rating", "feedback"],
                confidence_threshold=0.75,
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat()
            ),
            QuestionTemplate(
                id="recommendation_yesno",
                template_text="Would you recommend this product to others?",
                category="feedback",
                question_type="yes_no",
                response_strategy={
                    "generation_method": "persona_based",
                    "positive_bias": 0.7
                },
                examples=[
                    "Would you recommend us to a friend?",
                    "Are you likely to recommend this service?",
                    "Would you suggest this to others?"
                ],
                tags=["recommendation", "referral", "feedback"],
                confidence_threshold=0.8,
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat()
            )
        ]
        
        for template in default_templates:
            self.add_template(template)
        
        # Save default templates
        self.save_templates()
        
        self.logger.info(f"Initialized {len(default_templates)} default templates")

