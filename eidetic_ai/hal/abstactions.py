from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class AbstractionLevel:
    level: int
    name: str
    description: str
    examples: List[Dict[str, str]] = field(default_factory=list)

# Define the hierarchy for our kinematics domain
HIERARCHY = [
    AbstractionLevel(
        level=0,
        name="Factual Definition",
        description="Correctness of individual definitions, formulas, and properties.",
        examples=[
            {"prompt": "What is the definition of velocity?", "ideal_concepts": ["displacement", "time", "rate of change", "vector"]},
            {"prompt": "What are the units of acceleration?", "ideal_concepts": ["acceleration", "time", "distance"]},
        ]
    ),
    AbstractionLevel(
        level=1,
        name="Procedural Application",
        description="Correct application of a single formula or procedure in a simple context.",
        examples=[
            {"prompt": "A car travels 100 meters in 10 seconds. What is its average speed?", "ideal_concepts": ["distance", "time", "speed"]},
            {"prompt": "An object accelerates from rest at 2 m/s^2. What is its velocity after 5 seconds?", "ideal_concepts": ["acceleration", "initial state", "time", "velocity"]},
        ]
    ),
    AbstractionLevel(
        level=2,
        name="Conceptual Modeling",
        description="Integration of multiple concepts to explain a phenomenon or solve a complex problem.",
        examples=[
            {"prompt": "Explain why a thrown ball follows a parabolic path using the concepts of velocity and acceleration.", "ideal_concepts": ["velocity", "acceleration", "gravity", "vector", "projectile motion"]},
            {"prompt": "How does constant acceleration affect an object's displacement over time?", "ideal_concepts": ["constant acceleration", "displacement", "time", "velocity", "initial state", "integral"]},
        ]
    ),
]

def get_level_by_name(name: str) -> AbstractionLevel:
    for level in HIERARCHY:
        if level.name == name:
            return level
    raise ValueError(f"No abstraction level named '{name}' found.")