"""
Dynamic Persona Manager
Manages the creation, evolution, and selection of personas based on survey context.
"""

import json
import logging
import random
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class AgeGroup(Enum):
    """Age group enumeration."""
    TEEN = "13-19"
    YOUNG_ADULT = "20-29"
    ADULT = "30-44"
    MIDDLE_AGED = "45-59"
    SENIOR = "60+"


class IncomeLevel(Enum):
    """Income level enumeration."""
    LOW = "under_30k"
    LOWER_MIDDLE = "30k-60k"
    MIDDLE = "60k-100k"
    UPPER_MIDDLE = "100k-150k"
    HIGH = "150k+"


class EducationLevel(Enum):
    """Education level enumeration."""
    HIGH_SCHOOL = "high_school"
    SOME_COLLEGE = "some_college"
    BACHELORS = "bachelors"
    MASTERS = "masters"
    DOCTORATE = "doctorate"


@dataclass
class Persona:
    """Represents a dynamically generated persona."""
    persona_id: str
    name: str
    age: int
    age_group: str
    gender: str
    location: str
    occupation: str
    education: str
    income_level: str
    interests: List[str]
    values: List[str]
    personality_traits: List[str]
    purchase_habits: Dict[str, Any]
    tech_savviness: float  # 0.0 to 1.0
    social_media_usage: Dict[str, float]  # Platform -> usage frequency
    created_at: float
    last_updated: float
    survey_count: int = 0
    success_rate: float = 0.0
    evolution_history: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.evolution_history is None:
            self.evolution_history = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert persona to dictionary."""
        data = asdict(self)
        data['evolution_history'] = self.evolution_history
        return data

    def get_summary(self) -> str:
        """Get a human-readable summary of the persona."""
        return (
            f"{self.name}, {self.age} years old {self.gender} from {self.location}. "
            f"Works as a {self.occupation} with {self.education} education. "
            f"Interested in {', '.join(self.interests[:3])}. "
            f"Tech-savvy level: {self.tech_savviness:.1%}."
        )


class PersonaManager:
    """
    Manages dynamic persona creation, evolution, and selection based on survey context.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Persona Manager.
        
        Args:
            config: Configuration dictionary with persona settings.
        """
        self.config = config or {}
        self.logger = logging.getLogger("persona_manager")
        self.personas: Dict[str, Persona] = {}
        self.persona_pool: List[str] = []  # IDs of available personas
        self.context_history: List[Dict[str, Any]] = []

    def generate_persona(self, survey_context: Dict[str, Any]) -> Persona:
        """
        Dynamically generate a persona based on survey context.
        
        Args:
            survey_context: Dictionary containing survey metadata (type, topic, target_demographic, etc.)
            
        Returns:
            A newly generated Persona object.
        """
        persona_id = f"persona_{int(time.time() * 1000)}"
        
        # Extract context clues
        survey_type = survey_context.get("type", "general")
        topic = survey_context.get("topic", "general")
        target_demographic = survey_context.get("target_demographic", {})
        
        # Generate persona attributes based on context
        age_group = self._select_age_group(target_demographic)
        age = self._generate_age(age_group)
        gender = self._select_gender(target_demographic)
        location = self._select_location(target_demographic)
        
        # Generate occupation and education based on survey type
        occupation = self._select_occupation(survey_type, topic)
        education = self._select_education(occupation)
        income_level = self._select_income(occupation, education)
        
        # Generate interests and values based on survey topic
        interests = self._generate_interests(topic, age_group)
        values = self._generate_values(survey_type)
        personality_traits = self._generate_personality_traits()
        
        # Generate purchase habits
        purchase_habits = self._generate_purchase_habits(interests, income_level)
        
        # Tech savviness based on age and occupation
        tech_savviness = self._calculate_tech_savviness(age_group, occupation)
        
        # Social media usage
        social_media_usage = self._generate_social_media_usage(age_group)
        
        # Generate a realistic name
        name = self._generate_name(gender, age_group)
        
        persona = Persona(
            persona_id=persona_id,
            name=name,
            age=age,
            age_group=age_group,
            gender=gender,
            location=location,
            occupation=occupation,
            education=education,
            income_level=income_level,
            interests=interests,
            values=values,
            personality_traits=personality_traits,
            purchase_habits=purchase_habits,
            tech_savviness=tech_savviness,
            social_media_usage=social_media_usage,
            created_at=time.time(),
            last_updated=time.time()
        )
        
        # Store the persona
        self.personas[persona_id] = persona
        self.persona_pool.append(persona_id)
        
        self.logger.info(f"Generated persona {persona_id}: {persona.get_summary()}")
        return persona

    def evolve_persona(self, persona_id: str, feedback: Dict[str, Any]) -> Persona:
        """
        Evolve a persona based on feedback from survey interactions.
        
        Args:
            persona_id: ID of the persona to evolve.
            feedback: Feedback dictionary containing success metrics, user feedback, etc.
            
        Returns:
            The evolved Persona object.
        """
        if persona_id not in self.personas:
            self.logger.warning(f"Persona {persona_id} not found.")
            return None
        
        persona = self.personas[persona_id]
        
        # Record evolution event
        evolution_event = {
            "timestamp": time.time(),
            "feedback": feedback,
            "changes": {}
        }
        
        # Update survey count and success rate
        persona.survey_count += 1
        success = feedback.get("success", False)
        if success:
            persona.success_rate = (
                (persona.success_rate * (persona.survey_count - 1) + 1) / persona.survey_count
            )
        else:
            persona.success_rate = (
                (persona.success_rate * (persona.survey_count - 1)) / persona.survey_count
            )
        
        # Adjust interests based on feedback
        if "topic_feedback" in feedback:
            topic_feedback = feedback["topic_feedback"]
            if topic_feedback.get("relevant", False):
                for interest in topic_feedback.get("interests", []):
                    if interest not in persona.interests:
                        persona.interests.append(interest)
                        evolution_event["changes"]["interests"] = persona.interests
        
        # Adjust personality traits based on response patterns
        if "response_patterns" in feedback:
            patterns = feedback["response_patterns"]
            if patterns.get("conservative", False):
                if "cautious" not in persona.personality_traits:
                    persona.personality_traits.append("cautious")
                    evolution_event["changes"]["personality_traits"] = persona.personality_traits
            if patterns.get("adventurous", False):
                if "adventurous" not in persona.personality_traits:
                    persona.personality_traits.append("adventurous")
                    evolution_event["changes"]["personality_traits"] = persona.personality_traits
        
        # Update tech savviness if feedback indicates tech issues
        if "tech_issues" in feedback and feedback["tech_issues"]:
            persona.tech_savviness = max(0.0, persona.tech_savviness - 0.05)
            evolution_event["changes"]["tech_savviness"] = persona.tech_savviness
        
        persona.last_updated = time.time()
        persona.evolution_history.append(evolution_event)
        
        self.logger.info(f"Evolved persona {persona_id}. Success rate: {persona.success_rate:.2%}")
        return persona

    def select_best_persona(self, survey_context: Dict[str, Any]) -> Persona:
        """
        Select the best persona from the pool for a given survey context.
        
        Args:
            survey_context: Dictionary containing survey metadata.
            
        Returns:
            The best matching Persona object, or a newly generated one if none are suitable.
        """
        if not self.persona_pool:
            self.logger.info("No personas in pool, generating a new one.")
            return self.generate_persona(survey_context)
        
        # Score each persona based on context fit
        scores = {}
        for persona_id in self.persona_pool:
            persona = self.personas[persona_id]
            score = self._calculate_context_fit(persona, survey_context)
            scores[persona_id] = score
        
        # Select the persona with the highest score
        best_persona_id = max(scores, key=scores.get)
        best_persona = self.personas[best_persona_id]
        
        self.logger.info(f"Selected persona {best_persona_id} with fit score {scores[best_persona_id]:.2f}")
        return best_persona

    def _select_age_group(self, target_demographic: Dict[str, Any]) -> str:
        """Select an age group based on target demographic."""
        if "age_group" in target_demographic:
            return target_demographic["age_group"]
        return random.choice([ag.value for ag in AgeGroup])

    def _generate_age(self, age_group: str) -> int:
        """Generate a specific age within an age group."""
        age_ranges = {
            AgeGroup.TEEN.value: (13, 19),
            AgeGroup.YOUNG_ADULT.value: (20, 29),
            AgeGroup.ADULT.value: (30, 44),
            AgeGroup.MIDDLE_AGED.value: (45, 59),
            AgeGroup.SENIOR.value: (60, 80)
        }
        min_age, max_age = age_ranges.get(age_group, (18, 65))
        return random.randint(min_age, max_age)

    def _select_gender(self, target_demographic: Dict[str, Any]) -> str:
        """Select gender based on target demographic."""
        if "gender" in target_demographic:
            return target_demographic["gender"]
        return random.choice(["Male", "Female", "Non-binary"])

    def _select_location(self, target_demographic: Dict[str, Any]) -> str:
        """Select a location based on target demographic."""
        if "location" in target_demographic:
            return target_demographic["location"]
        
        us_cities = [
            "New York, NY", "Los Angeles, CA", "Chicago, IL", "Houston, TX",
            "Phoenix, AZ", "Philadelphia, PA", "San Antonio, TX", "San Diego, CA",
            "Dallas, TX", "San Jose, CA", "Austin, TX", "Jacksonville, FL",
            "Seattle, WA", "Denver, CO", "Boston, MA", "Miami, FL"
        ]
        return random.choice(us_cities)

    def _select_occupation(self, survey_type: str, topic: str) -> str:
        """Select an occupation based on survey type and topic."""
        occupation_map = {
            "product_feedback": ["Product Manager", "Designer", "Software Engineer", "Marketing Manager"],
            "consumer_behavior": ["Retail Manager", "Sales Associate", "Business Analyst", "Entrepreneur"],
            "health_wellness": ["Healthcare Worker", "Fitness Trainer", "Nutritionist", "Nurse"],
            "technology": ["Software Engineer", "IT Manager", "Data Scientist", "Tech Support"],
            "finance": ["Financial Analyst", "Accountant", "Investment Advisor", "Banker"],
            "general": ["Office Worker", "Consultant", "Manager", "Analyst", "Specialist"]
        }
        
        occupations = occupation_map.get(survey_type, occupation_map["general"])
        return random.choice(occupations)

    def _select_education(self, occupation: str) -> str:
        """Select education level based on occupation."""
        education_map = {
            "Software Engineer": EducationLevel.BACHELORS.value,
            "Data Scientist": EducationLevel.MASTERS.value,
            "Retail Manager": EducationLevel.SOME_COLLEGE.value,
            "Healthcare Worker": EducationLevel.BACHELORS.value,
            "Nurse": EducationLevel.BACHELORS.value,
            "Financial Analyst": EducationLevel.BACHELORS.value,
        }
        return education_map.get(occupation, random.choice([e.value for e in EducationLevel]))

    def _select_income(self, occupation: str, education: str) -> str:
        """Select income level based on occupation and education."""
        income_map = {
            "Software Engineer": IncomeLevel.UPPER_MIDDLE.value,
            "Data Scientist": IncomeLevel.UPPER_MIDDLE.value,
            "Retail Manager": IncomeLevel.MIDDLE.value,
            "Healthcare Worker": IncomeLevel.MIDDLE.value,
            "Financial Analyst": IncomeLevel.UPPER_MIDDLE.value,
        }
        return income_map.get(occupation, IncomeLevel.MIDDLE.value)

    def _generate_interests(self, topic: str, age_group: str) -> List[str]:
        """Generate interests based on survey topic and age group."""
        topic_interests = {
            "product_feedback": ["Technology", "Innovation", "Product Design"],
            "consumer_behavior": ["Shopping", "Brands", "Deals"],
            "health_wellness": ["Fitness", "Nutrition", "Mental Health"],
            "technology": ["Gadgets", "Software", "AI"],
            "finance": ["Investing", "Personal Finance", "Savings"],
            "general": ["Entertainment", "Travel", "Food"]
        }
        
        base_interests = topic_interests.get(topic, topic_interests["general"])
        age_interests = {
            AgeGroup.TEEN.value: ["Social Media", "Gaming", "Music"],
            AgeGroup.YOUNG_ADULT.value: ["Career", "Travel", "Fitness"],
            AgeGroup.ADULT.value: ["Family", "Home Improvement", "Hobbies"],
            AgeGroup.MIDDLE_AGED.value: ["Retirement", "Health", "Grandchildren"],
            AgeGroup.SENIOR.value: ["Travel", "Gardening", "Grandchildren"]
        }
        
        interests = base_interests + age_interests.get(age_group, [])
        return list(set(interests))[:5]  # Return up to 5 unique interests

    def _generate_values(self, survey_type: str) -> List[str]:
        """Generate core values based on survey type."""
        values_map = {
            "product_feedback": ["Quality", "Innovation", "Sustainability"],
            "consumer_behavior": ["Value for Money", "Convenience", "Brand Loyalty"],
            "health_wellness": ["Health", "Wellness", "Natural Products"],
            "technology": ["Progress", "Efficiency", "Cutting-edge"],
            "finance": ["Security", "Growth", "Stability"],
            "general": ["Honesty", "Family", "Community"]
        }
        return values_map.get(survey_type, values_map["general"])

    def _generate_personality_traits(self) -> List[str]:
        """Generate personality traits."""
        traits = ["Curious", "Thoughtful", "Practical", "Analytical", "Creative", "Honest", "Friendly"]
        return random.sample(traits, k=random.randint(3, 5))

    def _generate_purchase_habits(self, interests: List[str], income_level: str) -> Dict[str, Any]:
        """Generate purchase habits based on interests and income."""
        return {
            "frequency": random.choice(["daily", "weekly", "monthly"]),
            "average_spending": {"low": 50, "medium": 150, "high": 500}.get(income_level, 150),
            "preferred_channels": random.sample(["online", "in-store", "mobile app"], k=2),
            "brand_loyalty": random.choice(["high", "medium", "low"])
        }

    def _calculate_tech_savviness(self, age_group: str, occupation: str) -> float:
        """Calculate tech savviness based on age and occupation."""
        age_factor = {
            AgeGroup.TEEN.value: 0.9,
            AgeGroup.YOUNG_ADULT.value: 0.8,
            AgeGroup.ADULT.value: 0.6,
            AgeGroup.MIDDLE_AGED.value: 0.4,
            AgeGroup.SENIOR.value: 0.2
        }.get(age_group, 0.5)
        
        occupation_factor = 0.9 if "Engineer" in occupation or "Analyst" in occupation else 0.5
        return min(1.0, (age_factor + occupation_factor) / 2)

    def _generate_social_media_usage(self, age_group: str) -> Dict[str, float]:
        """Generate social media usage patterns based on age."""
        platforms = ["Facebook", "Instagram", "Twitter", "TikTok", "LinkedIn"]
        usage = {}
        
        if age_group == AgeGroup.TEEN.value:
            usage = {"TikTok": 0.9, "Instagram": 0.8, "Snapchat": 0.7, "Twitter": 0.3}
        elif age_group == AgeGroup.YOUNG_ADULT.value:
            usage = {"Instagram": 0.8, "Twitter": 0.6, "LinkedIn": 0.5, "TikTok": 0.4}
        elif age_group == AgeGroup.ADULT.value:
            usage = {"Facebook": 0.7, "LinkedIn": 0.6, "Instagram": 0.4, "Twitter": 0.2}
        elif age_group == AgeGroup.MIDDLE_AGED.value:
            usage = {"Facebook": 0.8, "LinkedIn": 0.5, "Instagram": 0.2}
        else:  # Senior
            usage = {"Facebook": 0.6, "LinkedIn": 0.1}
        
        return usage

    def _generate_name(self, gender: str, age_group: str) -> str:
        """Generate a realistic name based on gender and age."""
        first_names = {
            "Male": ["John", "Michael", "David", "James", "Robert", "William", "Richard", "Joseph"],
            "Female": ["Mary", "Patricia", "Jennifer", "Linda", "Barbara", "Elizabeth", "Susan", "Jessica"],
            "Non-binary": ["Alex", "Casey", "Morgan", "Riley", "Jordan", "Taylor", "Avery", "Quinn"]
        }
        
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
                      "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson"]
        
        first_name = random.choice(first_names.get(gender, first_names["Male"]))
        last_name = random.choice(last_names)
        return f"{first_name} {last_name}"

    def _calculate_context_fit(self, persona: Persona, survey_context: Dict[str, Any]) -> float:
        """
        Calculate how well a persona fits the survey context.
        
        Args:
            persona: The persona to evaluate.
            survey_context: The survey context.
            
        Returns:
            A fit score between 0.0 and 1.0.
        """
        score = 0.0
        weight_sum = 0.0
        
        # Check age group match
        if "target_demographic" in survey_context:
            target = survey_context["target_demographic"]
            if "age_group" in target and target["age_group"] == persona.age_group:
                score += 0.3
                weight_sum += 0.3
        
        # Check interest match
        if "topic" in survey_context:
            topic = survey_context["topic"]
            if any(topic.lower() in interest.lower() for interest in persona.interests):
                score += 0.2
                weight_sum += 0.2
        
        # Prefer personas with higher success rates
        score += persona.success_rate * 0.3
        weight_sum += 0.3
        
        # Prefer personas that haven't been used recently
        time_since_use = time.time() - persona.last_updated
        recency_score = min(1.0, time_since_use / 3600)  # Max score after 1 hour
        score += recency_score * 0.2
        weight_sum += 0.2
        
        return score / weight_sum if weight_sum > 0 else 0.5

