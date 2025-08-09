from dataclasses import dataclass
from typing import List, Callable, Any

# --- DSL Primitives ---

@dataclass
class Teach:
    """The most basic action: teach a specific set of concepts."""
    concepts: frozenset

@dataclass
class IfMastery:
    """A conditional branch based on student's knowledge."""
    condition_concepts: frozenset
    then_block: List[Any]
    else_block: List[Any]

@dataclass
class RepeatUntil:
    """A loop that continues until a student masters a concept set."""
    condition_concepts: frozenset
    body_block: List[Any]
    max_reps: int = 5 # Safety break

# --- The Interpreter ---

class ProgramInterpreter:
    """
    Executes a curriculum-program on a simulated student.
    """
    def __init__(self, student, all_concepts):
        self.student = student
        self.all_concepts = all_concepts
        self.trace = []

    def run(self, program_block: List[Any]):
        """Recursively executes a block of program statements."""
        for stmt in program_block:
            if isinstance(stmt, Teach):
                self.student.receive_lesson(stmt.concepts)
                self.trace.append(f"TEACH({', '.join(sorted(list(stmt.concepts)))})")
            
            elif isinstance(stmt, IfMastery):
                # Check if all condition concepts are mastered (knowledge > 0.8)
                indices = [self.student.concept_map[c] for c in stmt.condition_concepts]
                is_mastered = all(self.student.knowledge_vector[i] > 0.8 for i in indices)
                
                if is_mastered:
                    self.trace.append(f"IF_MASTERY({', '.join(sorted(list(stmt.condition_concepts)))}) -> THEN")
                    self.run(stmt.then_block)
                else:
                    self.trace.append(f"IF_MASTERY({', '.join(sorted(list(stmt.condition_concepts)))}) -> ELSE")
                    self.run(stmt.else_block)

            elif isinstance(stmt, RepeatUntil):
                for i in range(stmt.max_reps):
                    indices = [self.student.concept_map[c] for c in stmt.condition_concepts]
                    is_mastered = all(self.student.knowledge_vector[i] > 0.8 for i in indices)
                    
                    if is_mastered:
                        self.trace.append(f"REPEAT_UNTIL({', '.join(sorted(list(stmt.condition_concepts)))}) -> DONE")
                        break
                    
                    self.trace.append(f"REPEAT_UNTIL({', '.join(sorted(list(stmt.condition_concepts)))}) -> LOOP {i+1}")
                    self.run(stmt.body_block)
        return self.trace