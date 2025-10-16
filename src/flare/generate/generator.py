import asyncio
import json
import logging
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import litellm

litellm.add_function_to_prompt = True

from flare.complete import safe_completion
from flare.schema import (
    Generation,
    GenerationParams,
    ModelConfig,
    ModelOutputs,
    Output,
    OutputChoice,
    OutputUsage,
    Sample,
    SampleWithOutputs,
)


async def generate_output(
    generation: Generation,
    model: str,
    nb_try: int,
    **model_params,
) -> Output:
    params = generation.params.model_dump()
    if params["tools"] is None:
        params.pop("tools")
    else:
        # this is required because of side effect when using litellm with gemini models
        params["tools"] = deepcopy(params["tools"])

    if not litellm.supports_function_calling(model) and "tools" in params:
        params["functions_unsupported_model"] = params.pop("tools")
    params.update(model_params)

    response = await safe_completion(
        model_name=model,
        messages=[q.model_dump() for q in generation.messages],
        nb_try=nb_try,
        **params,
    )
    return Output(
        id=generation.id,
        choices=[
            OutputChoice(
                finish_reason=response.choices[0].finish_reason,
                message=response.choices[0].message,
                index=0,
            )
        ],
        created=datetime.fromtimestamp(response.created).isoformat(),
        usage=OutputUsage.model_validate(
            {
                **response.model_dump()["usage"],
                **(
                    {"cost": response._hidden_params["response_cost"]}
                    if not response.model_dump()["usage"].get("cost")
                    else {}
                ),
            }
        ),
        raw_responses=[response.model_dump(mode="json")],
    )


async def generate(
    run_name: str,
    model: ModelConfig,
    sample: Sample,
    logger: logging.Logger,
) -> SampleWithOutputs:
    model_params = model.model_dump()
    model_params.pop("name")
    model_params.pop("parallelism")

    model_name = model_params.pop("litellm_model")

    id_sample = sample.id
    target_path = (
        Path(run_name)
        / "generate"
        / model_name.replace("/", "_").replace(" ", "_")
        / (str(id_sample) + ".json")
    )

    logger.info("Checking path : %s", str(target_path))

    if target_path.is_file():
        # Return existing sample
        logger.info("Skipping path : %s", str(target_path))
        data = json.loads(target_path.read_text())
        return SampleWithOutputs.model_validate(data)

    target_path.parent.mkdir(parents=True, exist_ok=True)

    outputs = []
    total_fails = 0

    # TODO: this could be improved, but not necessary ?
    for question in sample.generations:
        logger.debug("Acompletion on question : %s", str(question))

        try:
            tasks = []
            # We use a task group to start all generation on a question at once
            async with asyncio.TaskGroup() as tg:
                for _ in range(question.num_repeats):
                    tasks.append(
                        tg.create_task(
                            generate_output(
                                generation=question,
                                model=model_name,
                                **model_params,
                            )
                        )
                    )
            # Now we are out of the group, so all tasked are finished or failed
            answers: Output = sum([t.result() for t in tasks])
        except Exception as e:
            logger.warning("Error while generating", exc_info=True)
            error = str(e) + traceback.format_exc()
            sample_dict = sample.model_dump(mode="json")
            sample_dict["error"] = error
            error_path = (
                Path(run_name)
                / "error"
                / "generate"
                / model_name.replace("/", "_").replace(" ", "_")
                / (str(id_sample) + ".json")
            )
            error_path.parent.mkdir(parents=True, exist_ok=True)
            error_path.write_text(json.dumps(sample_dict))
            return

        outputs.append(answers)

    sample_with_outputs = SampleWithOutputs(
        sample=sample,
        model_outputs=ModelOutputs(model=model_name, outputs=outputs),
    )
    target_path.write_text(sample_with_outputs.model_dump_json())
    return sample_with_outputs
