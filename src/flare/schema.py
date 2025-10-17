import os
import uuid
from typing import Any, Literal, Optional

from litellm import Message
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_GENERATOR_NB_TRY = int(os.getenv("DEFAULT_GENERATOR_NB_TRY", "5"))
DEFAULT_GENERATOR_CONCURRENCY = int(os.getenv("DEFAULT_GENERATOR_CONCURRENCY", "8"))
DEFAULT_SCORER_CONCURRENCY = int(os.getenv("DEFAULT_SCORER_CONCURRENCY", "10"))


class FlareModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class GenerationParams(FlareModel):
    temperature: float = Field(0.0, ge=0.0)
    tools: Any | None = Field(None)
    max_tokens: int = Field(4096)
    extra_body: Optional[dict[str, Any]] = Field(default_factory=dict)


# TODO : improve tool handling when needed
# https://docs.litellm.ai/docs/completion/input
# tools: array (optional) - A list of tools the model may call. Currently, only functions are supported as a tool. Use this to provide a list of functions the model may generate JSON inputs for.

# type: string - The type of the tool. Currently, only function is supported.

# function: object - Required.

# tool_choice: string or object (optional) - Controls which (if any) function is called by the model. none means the model will not call a function and instead generates a message. auto means the model can pick between generating a message or calling a function. Specifying a particular function via {"type: "function", "function": {"name": "my_function"}} forces the model to call that function.

# none is the default when no functions are present. auto is the default if functions are present.


class Generation(FlareModel):
    id: uuid.UUID
    type: Literal["chat_completion"]
    messages: list[Message]
    params: Optional[GenerationParams] = Field(default_factory=GenerationParams)
    metadata: dict[str, Any] = Field(default_factory=dict)
    num_repeats: Optional[int] = Field(1)


class EvaluationData(FlareModel):
    scorer: str
    data: dict[str, Any] = Field(default_factory=dict)


class Sample(FlareModel):
    id: uuid.UUID
    module: str
    task: str
    language: str
    generations: list[Generation]
    metadata: dict[str, Any] = Field(default_factory=dict)
    evaluation: EvaluationData


class OutputUsage(FlareModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float | None = None

    def __add__(self, other: "OutputUsage"):
        # TODO: add small unit test on this
        return self.__class__(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cost=self.cost + other.cost if self.cost and other.cost else None,
        )


class OutputChoice(FlareModel):
    finish_reason: Literal[
        "stop", "length", "function_call", "content_filter", "tool_calls", "null"
    ]
    index: int
    message: Message


class Output(FlareModel):
    id: uuid.UUID
    choices: list[OutputChoice]
    created: str
    usage: OutputUsage
    raw_responses: list[dict[str, Any]]

    def __radd__(self, other: "Output"):
        if not isinstance(other, Output):
            return self

        return other.__add__(self)

    def __add__(self, other: "Output"):
        # TODO: add small unit test on this
        return self.__class__(
            id=self.id,
            choices=self.choices
            + [
                OutputChoice(
                    finish_reason=c.finish_reason,
                    message=c.message,
                    index=i + len(self.choices),
                )
                for i, c in enumerate(other.choices)
            ],
            created=self.created,  # maybe we should take the max of the two
            usage=self.usage + other.usage,
            raw_responses=self.raw_responses + other.raw_responses,
        )


class ModelOutputs(FlareModel):
    model: str
    outputs: list[Output]


class SampleWithOutputs(FlareModel):
    sample: Sample
    model_outputs: ModelOutputs


class ScorerOutput(FlareModel):
    score: float = Field(
        description="float between 0 and 1, 0 being the worst score, 1 being the best",
        le=1.0,
        ge=0.0,
    )
    details: dict[str, Any] = Field(
        default_factory=dict, description="optional details about the scoring process"
    )
    usage: dict[str, OutputUsage] = Field(default_factory=dict)

# TODO: would be better to have subclass of scorer, with custom details as pydantic model


class SampleOutputsWithScore(FlareModel):
    sample_with_outputs: SampleWithOutputs
    scoring: ScorerOutput


class ScorerParams(FlareModel):
    model_config = ConfigDict(extra="allow")
    
    temperature: float = Field(0.0, ge=0.0)
    max_tokens: int = Field(4096)
    n: int = Field(1)
    region: str | None = None


class ScorerModelConfig(FlareModel):
    model_config = ConfigDict(extra="allow")

    litellm_model: str
    weight: float
    generation_kwargs: ScorerParams = Field(default_factory=ScorerParams)


class ScorerConfig(FlareModel):
    parallelism: int = Field(default=DEFAULT_SCORER_CONCURRENCY)
    models: list[ScorerModelConfig]


class ModelConfig(FlareModel):
    model_config = ConfigDict(extra="allow")

    name: str
    litellm_model: str
    parallelism: int = Field(default=DEFAULT_GENERATOR_CONCURRENCY)
    nb_try: int = Field(default=DEFAULT_GENERATOR_NB_TRY)
    region: str | None = None


class Config(FlareModel):
    models: list[ModelConfig]
    scorers: dict[str, ScorerConfig]
