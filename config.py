"""Judge type enums and helpers shared by llm_judges/deploy.py.

Training configuration has moved to configs/ — see configs/qwen3_4b_haiku.py.
"""

from enum import Enum


class JudgeType(str, Enum):
    STANDARD = "standard"                    # Without curriculum learning
    CURRICULUM_LEARNING = "cl"  # Curriculum learning
    NO_LLM = "no-llm"                       # Only use the structure score


class JudgeModelSize(str, Enum):
    QWEN3_4B = "Qwen/Qwen3-4B"
    QWEN3_30B = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    QWEN3_235B = "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"

    @property
    def model_name(self) -> str:
        return {
            self.QWEN3_4B: "qwen3-4b",
            self.QWEN3_30B: "qwen3-30b-a3b-instruct",
            self.QWEN3_235B: "qwen3-235b-a22b-instruct-fp8",
        }[self.value]

    @property
    def shorthand(self) -> str:
        return {
            self.QWEN3_4B: "4b",
            self.QWEN3_30B: "30b",
            self.QWEN3_235B: "235b",
        }[self.value]


def _judge_class_name(judge_type: JudgeType, judge_model_size: JudgeModelSize) -> str:
    """Generate Modal class name for a judge combo, e.g. 'Standard30b'."""
    type_part = "".join(part.capitalize() for part in judge_type.value.split("-"))
    return f"{type_part}{judge_model_size.shorthand.capitalize()}"
