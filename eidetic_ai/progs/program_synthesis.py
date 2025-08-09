import random
from typing import List, Any
from eidetic_ai.progs.dsl import Teach, IfMastery, RepeatUntil

class ProgrammaticTeacher:
    """
    Represents an individual teacher defined by a curriculum-program.
    This is the "genotype" that will be evolved.
    """
    def __init__(self, program: List[Any], all_concepts: List[str]):
        self.program = program
        self.all_concepts = all_concepts
        self.fitness = -1.0 # Will be evaluated by the simulation

    def __str__(self):
        # A simple pretty-printer for the program
        def print_block(block, indent=0):
            s = ""
            for stmt in block:
                s += " " * indent
                if isinstance(stmt, Teach):
                    s += f"Teach({', '.join(sorted(list(stmt.concepts)))})\n"
                elif isinstance(stmt, IfMastery):
                    s += f"IfMastery({', '.join(sorted(list(stmt.condition_concepts)))}):\n"
                    s += print_block(stmt.then_block, indent + 2)
                    s += " " * indent + "Else:\n"
                    s += print_block(stmt.else_block, indent + 2)
                elif isinstance(stmt, RepeatUntil):
                    s += f"RepeatUntil({', '.join(sorted(list(stmt.condition_concepts)))}):\n"
                    s += print_block(stmt.body_block, indent + 2)
            return s
        return print_block(self.program)

    def mutate(self):
        """Applies a random mutation to the program."""
        # This is a simple mutation operator. A real system would have more.
        # It finds a random 'Teach' block and adds/removes a random concept.
        
        # Flatten the program to find all Teach statements
        all_teach_stmts = []
        def find_teach(block):
            for stmt in block:
                if isinstance(stmt, Teach):
                    all_teach_stmts.append(stmt)
                elif isinstance(stmt, (IfMastery, RepeatUntil)):
                    if hasattr(stmt, 'then_block'): find_teach(stmt.then_block)
                    if hasattr(stmt, 'else_block'): find_teach(stmt.else_block)
                    if hasattr(stmt, 'body_block'): find_teach(stmt.body_block)

        find_teach(self.program)
        if not all_teach_stmts: return

        # Pick one and mutate it
        stmt_to_mutate = random.choice(all_teach_stmts)
        mutable_concepts = set(stmt_to_mutate.concepts)
        
        if random.random() > 0.5: # Add a concept
            possible_additions = [c for c in self.all_concepts if c not in mutable_concepts]
            if possible_additions:
                mutable_concepts.add(random.choice(possible_additions))
        else: # Remove a concept
            if len(mutable_concepts) > 1:
                mutable_concepts.remove(random.choice(list(mutable_concepts)))
        
        stmt_to_mutate.concepts = frozenset(mutable_concepts)

def crossover(parent1: ProgrammaticTeacher, parent2: ProgrammaticTeacher) -> ProgrammaticTeacher:
    """
    Creates a new child program by swapping a random subtree between two parents.
    """
    # This is a very simple crossover. In practice, you'd use more structured
    # genetic programming operators. For this demo, we'll just pick one of the parents.
    child_program = random.choice([parent1.program, parent2.program])
    return ProgrammaticTeacher(child_program, parent1.all_concepts)


def generate_random_program(all_concepts: List[str], max_depth=2, current_depth=0) -> List[Any]:
    """Generates a small, random curriculum-program."""
    if current_depth >= max_depth:
        # At max depth, just teach a small set of random concepts
        num_concepts = random.randint(1, 4)
        concepts = frozenset(random.sample(all_concepts, num_concepts))
        return [Teach(concepts)]

    num_stmts = random.randint(1, 2)
    program = []
    for _ in range(num_stmts):
        stmt_type = random.choice([Teach, IfMastery])
        if stmt_type == Teach:
            num_concepts = random.randint(2, 5)
            concepts = frozenset(random.sample(all_concepts, num_concepts))
            program.append(Teach(concepts))
        elif stmt_type == IfMastery:
            num_cond_concepts = random.randint(1, 2)
            cond_concepts = frozenset(random.sample(all_concepts, num_cond_concepts))
            then_block = generate_random_program(all_concepts, max_depth, current_depth + 1)
            else_block = generate_random_program(all_concepts, max_depth, current_depth + 1)
            program.append(IfMastery(cond_concepts, then_block, else_block))
    return program