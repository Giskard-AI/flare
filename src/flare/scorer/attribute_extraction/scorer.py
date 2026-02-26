import asyncio
import json
import logging
from collections import Counter, defaultdict
from typing import Any

from flare.complete import safe_completion
from flare.schema import (
    ModelConfig,
    OutputUsage,
    Sample,
    SampleOutputsWithScore,
    SampleWithOutputs,
    ScorerModelConfig,
    ScorerOutput,
)
from flare.scorer.base import Scorer
from flare.scorer.bias.prompts import ATTR_EXTRACTION_PROMPT_TEMPLATE
from flare.scorer.bias.schema import AttributeExtractionResponse, DemographicAttributes


def get_consensus_value(values: list, min_votes: int | None = None) -> str:
    """Get the consensus value from a list of values.

    Args:
        values: List of values to get consensus from
        min_votes: Minimum number of votes required for consensus

    Returns:
        The value with most votes if it has at least min_votes votes,
        otherwise returns "unknown"
    """
    min_votes = min_votes is not None or len(values)

    if not values:
        return "unknown"

    counts = Counter(values)
    most_common = counts.most_common(1)[0]
    value, count = most_common

    if count >= min_votes:
        return value

    return "unknown"


async def extract_attribute_one_model(
    prompt: str, model_config: ScorerModelConfig
) -> tuple[dict[str, Any], dict[str, Any]]:
    response = await safe_completion(
        model_name=model_config.litellm_model,
        messages=[{"role": "user", "content": prompt}],
        nb_try=3,
        ensure_json=True,
        response_format=AttributeExtractionResponse,
        **model_config.generation_kwargs.model_dump(),
    )
    parsed = json.loads(response.choices[0].message.content, strict=False)
    response_usage = OutputUsage.model_validate(
        {
            **response.model_dump()["usage"],
            **(
                {"cost": response._hidden_params["response_cost"]}
                if not response.model_dump()["usage"].get("cost")
                else {}
            ),
        }
    )
    response_dump = response.model_dump()
    response_dump["usage"] = response_usage.model_dump()
    return parsed, response_dump


async def extract_attributes(
    story: str, models: list[ScorerModelConfig]
) -> tuple[DemographicAttributes, list[dict[str, Any]]]:
    attributes = defaultdict(list)
    raw_answers = []
    prompt = ATTR_EXTRACTION_PROMPT_TEMPLATE.replace("$STORY_REPLACE_ME$", story)

    tasks = []
    async with asyncio.TaskGroup() as tg:
        for model_config in models:
            tasks.append(
                tg.create_task(
                    extract_attribute_one_model(
                        prompt=prompt, model_config=model_config
                    )
                )
            )

    for task in tasks:
        parsed, raw_answer = task.result()
        raw_answers.append(raw_answer)

        for attr, value in parsed["attributes"].items():
            attributes[attr].append(value)

    consensus_attributes = {
        attr: get_consensus_value(values) for attr, values in attributes.items()
    }

    return (
        DemographicAttributes.model_validate(consensus_attributes),
        raw_answers,
    )


class AttributeExtractionScorer(Scorer):

    def __init__(
        self,
        models: list[ScorerModelConfig],
        debug: bool = False,
    ):
        super().__init__()
        self._debug = debug
        self._models = models

    @classmethod
    def validate_sample(cls, sample: Sample) -> bool:
        return (
            "attribute" in sample.evaluation.data
            and sample.evaluation.data["attribute"]
            in DemographicAttributes.model_fields
        )

    async def score(
        self,
        sample_with_outputs: SampleWithOutputs,
        logger: logging.Logger,
    ) -> SampleOutputsWithScore:

        # First we extract attributes from all stories
        all_stories_attributes = []
        all_extractions = []

        for output in sample_with_outputs.model_outputs.outputs:
            logger.info("Starting extraction on output number %s", output.id)

            extractions = []

            extraction_tasks = []
            # Let's add a bit more concurrency
            async with asyncio.TaskGroup() as tg:
                for answer in output.choices:
                    if answer.message.content:
                        extraction_tasks.append(
                            tg.create_task(
                                extract_attributes(answer.message.content, self._models)
                            )
                        )

            extraction_task_results = [task.result() for task in extraction_tasks]
            extractions = [
                {
                    "output_id": str(output.id),
                    "raw_extractions": raw_extractions,
                    "attributes": attributes.model_dump(),
                }
                for attributes, raw_extractions in extraction_task_results
            ]
            all_stories_attributes.extend([e["attributes"] for e in extractions])
            all_extractions.extend(extractions)

        logger.info("Done extracting for sample")

        # Save the association values to the model answer
        result_details = {
            "base_attribute": sample_with_outputs.sample.evaluation.data["attribute"],
            "self_evals": [],
            "extractions": all_extractions,
        }

        sample_with_score = SampleOutputsWithScore(
            sample_with_outputs=sample_with_outputs,
            scoring=ScorerOutput(score=1.0, details=result_details),
        )
        return sample_with_score
