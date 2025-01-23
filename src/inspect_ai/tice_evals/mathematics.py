"""
Measuring Mathematical Problem Solving With the MATH Dataset

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora,
Steven Basart, Eric Tang, Dawn Song, Jacob Steinhardt
https://arxiv.org/abs/2103.03874

Based on:
1. https://github.com/openai/simple-evals/blob/main/math_eval.py
2. https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/minerva_math
3. https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math

# default run - zero shot
inspect eval mathematics.py

# run with fewshot samples
inspect eval mathematics.py -T fewshot=4

# run for specific levels and/or subjects
inspect eval mathematics.py -T levels=4,5 -T subjects=algebra

# run using a different grader model
inspect eval mathematics.py -T grader_model=openai/gpt-4
"""

from typing import Literal

from inspect_ai import Task, task, Epochs
from inspect_ai.dataset import hf_dataset
from inspect_ai.model import GenerateConfig, Model
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import (
    Solver,
    TaskState,
    generate,
    prompt_template,
    system_message,
)

from mathematics.utils import (
    filter_dataset,
    record_to_sample,
    sample_to_fewshot,
    score_helper,
)

# Few-shot prompt template partially based on https://arxiv.org/pdf/2206.14858 - Appendix D.2
SYSTEM_W_EXAMPLES_PROMPT_TEMPLATE = """
You will be asked to solve a math problem. Some examples of problems and solutions are provided below.

{examples}
""".strip()

# Setup for problem + instructions for providing answer with CoT
USER_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.
""".strip()

# Setup for problem + instructions for providing answer without CoT
USER_PROMPT_TEMPLATE_NO_CoT = """
Solve the following math problem. Your answer should be in the following format: "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to respond directly with your answer in the following format: "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.
""".strip()

# Add this constant at the top with other constants
NUM_EPOCHS = 5


@task
def math(
    levels: list[Literal[1, 2, 3, 4, 5]] = [1],
    subjects: list[str] = [],
    fewshot: int = 0,
    fewshot_seed: int = 42,
    test_seed: int = 42,
    grader_model: str | None = "google/gemini-1.5-flash",
    num_problems: int = 50,
) -> Task:
    """
    Inspect Task implementation for the MATH benchmark

    Args:
        levels (list[Literal[1,2,3,4,5]]): List of levels to filter on, 1 to 5.
        subjects (list[str]): List of subjects to filter on.
        fewshot (int): The number of fewshots to include
        fewshot_seed (int): The seed used when selecting fewshots
        test_seed (int): The seed used for selecting test problems
        grader_model (str): The model used to grade the samples (defaults to Gemini 1.5 Pro)
        num_problems (int): Number of problems to evaluate (default: 10)
    """
    dataset = hf_dataset(
        path="hendrycks/competition_math",
        split="test",
        trust=True,
        sample_fields=record_to_sample,
        auto_id=True,
        shuffle=True,
        seed=test_seed,
    )
    # First filter the dataset based on levels and/or subjects
    dataset = filter_dataset(dataset=dataset, levels=levels, subjects=subjects)
    # Then limit the number of problems using list slicing
    dataset = dataset[:num_problems]

    return Task(
        dataset=dataset,
        solver=math_solver(fewshot=fewshot, fewshot_seed=fewshot_seed),
        scorer=[
            expression_equivalance(model=grader_model),
            expression_exact_match(),
            expression_exact_match_sympy(),
        ],
        config=GenerateConfig(temperature=0.5),
        epochs=Epochs(NUM_EPOCHS, ["mean", "pass_at_1", "pass_at_2", "pass_at_5"]),
    )


@scorer(metrics=[accuracy(), stderr()])
def expression_equivalance(model: str | Model | None) -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        return await score_helper(
            state=state,
            target=target,
            model=model,
            exact_match=False,
        )

    return score


# Exact match using sympy based on: https://arxiv.org/pdf/2206.14858
@scorer(metrics=[accuracy(), stderr()])
def expression_exact_match_sympy() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        return await score_helper(
            state=state,
            target=target,
            exact_match=True,
            use_sympy=True,
        )

    return score


# Exact match based on:
# https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py#L36
@scorer(metrics=[accuracy(), stderr()])
def expression_exact_match() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        return await score_helper(
            state=state,
            target=target,
            exact_match=True,
            use_sympy=False,
        )

    return score


def math_solver(
    fewshot: int,
    fewshot_seed: int,
) -> list[Solver]:
    """Build solver for MATH task.

    Arguments:
        fewshot (int): Number of few shot examples to use.
        fewshot_seed (int): Random seed for sampling few shot examples.
    """
    solver = [prompt_template(USER_PROMPT_TEMPLATE), generate()]

    if fewshot:
        fewshot_samples = hf_dataset(
            path="hendrycks/competition_math",
            split="train",
            trust=True,
            sample_fields=record_to_sample,
            shuffle=True,
            seed=fewshot_seed,
            limit=fewshot,
        )
        solver.insert(
            0,
            system_message(
                SYSTEM_W_EXAMPLES_PROMPT_TEMPLATE.format(
                    examples="\n\n".join(
                        [sample_to_fewshot(sample=sample) for sample in fewshot_samples]
                    )
                )
            ),
        )

    return solver
