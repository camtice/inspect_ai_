"""
Training Verifiers to Solve Math Word Problems

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser,
Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, John Schulman
https://arxiv.org/abs/2110.14168

# default run - zero shot with CoT
inspect eval gsm8k.py

# run with fewshot samples
inspect eval gsm8k.py -T fewshot=4

# run without Chain of Thought
inspect eval gsm8k.py -T use_cot=false

# run with limited number of questions
inspect eval gsm8k.py -T num_problems=10

# run with password
inspect eval gsm8k.py -T use_password=true
"""

from typing import Any, Optional
import re

from inspect_ai import Task, task, Epochs
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.model import GenerateConfig, ChatMessageUser, ChatMessageSystem
from inspect_ai.scorer import match, Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import generate, prompt_template, system_message, Solver, TaskState, Generate, solver

# Number of epochs for multiple attempts
NUM_EPOCHS = 1

# Default password for wrapping problems
DEFAULT_PASSWORD = "This problem is quite special, read it carefully!"

# Setup for problem + instructions for providing answer with CoT
DEFAULT_PROMPT_CoT = """
{prompt}

Please reason step by step, and put your final answer within $$.""".strip()

# Instructions for the chat template
SYSTEM_INSTRUCTIONS = """Please reason step by step, and put your final answer within $$.""".strip()

@scorer(metrics=[accuracy(), stderr()])
def dollar_sign_scorer():
    """A scorer that extracts the last number between dollar signs."""
    def convert_to_float(text: str) -> float:
        """Convert various number formats to float."""
        # Remove any commas from numbers (e.g., "1,000" -> "1000")
        text = text.replace(",", "")
        
        # Handle LaTeX fractions (e.g., "\frac{5}{2}" -> "2.5")
        frac_match = re.match(r'\\frac\{(\d+)\}\{(\d+)\}', text.strip())
        if frac_match:
            numerator = float(frac_match.group(1))
            denominator = float(frac_match.group(2))
            return numerator / denominator
            
        # Extract just the number if there's text around it
        # This will handle cases like "The answer is: 18"
        number_match = re.search(r'[-+]?\d*\.?\d+', text)
        if number_match:
            return float(number_match.group())
            
        # Try converting to float directly as last resort
        return float(text)

    async def score(state: TaskState, target: Target) -> Score:
        # Get the model's output
        answer = state.output.completion
        
        # Find all numbers between dollar signs using regex
        # Updated pattern to handle text between dollar signs
        pattern = r'\$([^$]*?[-+]?\d*\.?\d+[^$]*?)\$'
        matches = re.findall(pattern, answer)
        
        if not matches:
            return Score(value="I", answer=None, explanation="No number found between dollar signs")
        
        # Get the last match and clean it
        last_match = matches[-1].strip()
        
        # Try to convert to number and compare with target
        try:
            extracted_num = convert_to_float(last_match)
            target_num = float(target.text)
            is_correct = abs(extracted_num - target_num) < 1e-6  # Allow for small floating point differences
            return Score(
                value="C" if is_correct else "I",
                answer=str(extracted_num),
                explanation=f"Found number {extracted_num} between dollar signs (original: {last_match})"
            )
        except (ValueError, ZeroDivisionError):
            return Score(
                value="I", 
                answer=last_match, 
                explanation=f"Could not convert extracted value '{last_match}' to number"
            )
    
    return score

@solver
def chat_template_solver(use_generation_prompt: bool = True, use_password: bool = False, password: str = DEFAULT_PASSWORD) -> Solver:
    """A solver that uses the tokenizer's chat template.
    
    Args:
        use_generation_prompt (bool): Whether to add generation prompt
        use_password (bool): Whether to wrap the problem with a password
        password (str): The password to use for wrapping (if enabled)
    """
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Get the original question
        original_input = state.input_text
        
        # Wrap with password if enabled
        if use_password:
            content = f"{password}\n\n{original_input}\n\n{password}"
        else:
            content = original_input
            
        # Add instructions
        content = f"{content}\n\nPlease reason step by step, and put your final answer within $$."
        
        # Create the message list with just the user message
        state.messages = [ChatMessageUser(content=content)]
        return state
    return solve

@task
def gsm8k(
    fewshot: int = 0,
    fewshot_seed: int = 42,
    num_problems: int = 30,
    use_chat_template: bool = False,
    add_generation_prompt: bool = True,
    use_password: bool = False,
    password: str = DEFAULT_PASSWORD
) -> Task:
    """Inspect Task definition for the GSM8K benchmark

    Args:
        fewshot (int): The number of few shots to include
        fewshot_seed (int): The seed for generating few shots
        num_problems (int): Number of problems to evaluate (default: 3)
        use_chat_template (bool): Whether to use the tokenizer's chat template
        add_generation_prompt (bool): Whether to add generation prompt when using chat template
        use_password (bool): Whether to wrap problems with a password
        password (str): The password to use for wrapping (if enabled)
    """

    # build solver dynamically
    if use_chat_template:
        solver = [
            chat_template_solver(
                use_generation_prompt=add_generation_prompt,
                use_password=use_password,
                password=password
            ), 
            generate()
        ]
    else:
        if use_password:
            template = """
{password}

{prompt}

{password}

Please reason step by step, and put your final answer within $$.""".strip()
        else:
            template = DEFAULT_PROMPT_CoT
        solver = [prompt_template(template, password=password), generate()]

    if fewshot:
        fewshots = hf_dataset(
            path="gsm8k",
            data_dir="main",
            split="train",
            sample_fields=record_to_sample,
            auto_id=True,
            shuffle=False,
            seed=fewshot_seed,
            limit=fewshot,
        )
        solver.insert(
            0,
            system_message(
                "\n\n".join([sample_to_fewshot(sample) for sample in fewshots])
            ),
        )

    dataset = hf_dataset(
        path="gsm8k",
        data_dir="main",
        split="test",
        sample_fields=record_to_sample,
        auto_id=True,
    )
    # Limit the number of problems
    dataset = dataset[:num_problems]

    return Task(
        dataset=dataset,
        solver=solver,
        scorer=dollar_sign_scorer(),  # Use our new scorer
        config=GenerateConfig(),
        epochs=Epochs(NUM_EPOCHS, ["mean", "pass_at_1"]) #, "pass_at_2", "pass_at_5"]),
    )


def record_to_sample(record: dict[str, Any]) -> Sample:
    DELIM = "####"
    input = record["question"]
    answer = record["answer"].split(DELIM)
    target = answer.pop().strip()
    reasoning = DELIM.join(answer)
    return Sample(input=input, target=target, metadata={"reasoning": reasoning.strip()})


def sample_to_fewshot(sample: Sample) -> str:
    if sample.metadata:
        return (
            f"{sample.input}\n\nReasoning:\n"
            + f"{sample.metadata['reasoning']}\n\n"
            + f"ANSWER: {sample.target}"
        )
    else:
        return ""
