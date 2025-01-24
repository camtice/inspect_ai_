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
"""

from typing import Any

from inspect_ai import Task, task, Epochs
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import match
from inspect_ai.solver import generate, prompt_template, system_message

# Number of epochs for multiple attempts
NUM_EPOCHS = 5

# Setup for problem + instructions for providing answer with CoT
USER_PROMPT_TEMPLATE = """
Solve the following math word problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.
""".strip()

# Setup for problem + instructions for providing answer without CoT
USER_PROMPT_TEMPLATE_NO_CoT = """
Solve the following math word problem. Your answer should be in the following format: "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to respond directly with your answer in the following format: "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.
""".strip()


@task
def gsm8k(
    fewshot: int = 0,
    fewshot_seed: int = 42,
    use_cot: bool = False,
    num_problems: int = 50,
) -> Task:
    """Inspect Task definition for the GSM8K benchmark

    Args:
        fewshot (int): The number of few shots to include
        fewshot_seed (int): The seed for generating few shots
        use_cot (bool): Whether to use Chain of Thought prompting
        num_problems (int): Number of problems to evaluate (default: 50)
    """
    # build solver dynamically (may or may not be doing fewshot)
    template = USER_PROMPT_TEMPLATE if use_cot else USER_PROMPT_TEMPLATE_NO_CoT
    solver = [prompt_template(template), generate()]

    if fewshot:
        fewshots = hf_dataset(
            path="gsm8k",
            data_dir="main",
            split="train",
            sample_fields=record_to_sample,
            auto_id=True,
            shuffle=True,
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
        scorer=match(numeric=True),
        config=GenerateConfig(temperature=0.5),
        epochs=Epochs(NUM_EPOCHS, ["mean", "pass_at_1", "pass_at_2", "pass_at_5"]),
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
