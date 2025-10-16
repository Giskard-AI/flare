import uuid
from functools import partial

from flare.schema import Sample


def phare_data_to_sample(
    sample_json: dict, module: str, task: str, scorer: str
) -> Sample:
    metadata = sample_json["metadata"].copy()
    metadata.pop("task_name", None)
    language = metadata.pop("language", None)
    eval_data = sample_json["evaluation_data"]
    eval_data.pop("task_name", None)

    if "template_type" in eval_data:
        metadata["template_type"] = eval_data.pop("template_type")
    if "template_name" in eval_data:
        metadata["template_name"] = eval_data.pop("template_name")

    return Sample(
        id=uuid.UUID(sample_json["id"]),
        module=module,
        task=task,
        language=language,
        generations=[
            {
                "id": uuid.uuid4(),
                "type": "chat_completion",
                "messages": sample_json["messages"],
            }
        ],
        metadata=metadata,
        evaluation={"scorer": scorer, "data": eval_data},
    )


def phare_debunking_to_sample(sample_json: dict) -> Sample:
    metadata = sample_json["metadata"].copy()
    language = metadata.pop("language", None)

    eval_args = sample_json["evaluation_data"]["args"]["eval_args"]
    criterion = eval_args.pop("criterion", None)
    context = eval_args.pop("context", None)

    metadata["prompt_level"] = eval_args.pop("prompt_level", None)
    metadata["category"] = eval_args.pop("task_name", None)

    return Sample(
        id=uuid.UUID(sample_json["id"]),
        module="hallucination",
        task="debunking",
        language=language,
        generations=[
            {
                "id": uuid.uuid4(),
                "type": "chat_completion",
                "messages": sample_json["messages"],
            }
        ],
        metadata=metadata,
        evaluation={
            "scorer": "debunking",
            "data": {"criterion": criterion, "context": context},
        },
    )


def phare_tools_to_sample(sample_json: dict) -> Sample:
    evaluation_data = sample_json["evaluation_data"]
    evaluation_data.pop("template_type", None)
    task_name = evaluation_data.pop("task_name", None)
    metadata = sample_json["metadata"]
    language = metadata.pop("language", None)
    metadata.pop("task_name", None)

    return Sample(
        id=uuid.UUID(sample_json["id"]),
        module="tools",
        task=task_name.split("/")[-1],
        language=language,
        generations=[
            {
                "id": uuid.uuid4(),
                "type": "chat_completion",
                "messages": sample_json["messages"],
                "params": {"tools": sample_json["tools"], "max_tokens": 1024},
            }
        ],
        metadata=metadata,
        evaluation={"scorer": "tools", "data": evaluation_data},
    )


def phare_story_generation_to_sample(sample_json: dict) -> Sample:
    metadata = sample_json["metadata"].copy()
    metadata.pop("task_name", None)
    language = metadata.pop("language", None)

    question_set = sample_json["question_set"]
    for q in question_set:
        q["metadata"].pop("task_name", None)
        q["metadata"].pop("language", None)

    return Sample(
        id=uuid.UUID(sample_json["id"]),
        module="biases",
        task="story_generation",
        language=language,
        generations=[
            {
                "id": q["id"],
                "type": "chat_completion",
                "messages": q["messages"],
                "metadata": q["metadata"],
                "params": {"temperature": sample_json["metadata"]["temperature"]},
                "num_repeats": sample_json["metadata"]["num_repeats"],
            }
            for q in sample_json["question_set"]
        ],
        metadata=metadata,
        evaluation={
            "scorer": "biases/story_generation",
            "data": {"attribute": sample_json["metadata"]["attribute"]},
        },
    )


def phare_jailbreak_to_sample(sample_json: dict) -> Sample:
    eval_data = sample_json["evaluation_data"].copy()
    task_name = eval_data.pop("task_name", None)
    submodule = task_name.split("/")[1]

    metadata = sample_json["metadata"]
    language = metadata.pop("language", None)

    return Sample(
        id=uuid.UUID(sample_json["id"]),
        module="jailbreak",
        task=submodule,
        language=language,
        generations=[
            {
                "id": uuid.uuid4(),
                "type": "chat_completion",
                "messages": sample_json["messages"],
            }
        ],
        metadata=metadata,
        evaluation={"scorer": task_name, "data": {"category": submodule, **eval_data}},
    )


IMPORTERS = {
    "biases/story_generation": phare_story_generation_to_sample,
    "hallucination/factuality": partial(
        phare_data_to_sample,
        module="hallucination",
        task="factuality",
        scorer="factuality",
    ),
    "hallucination/debunking": phare_debunking_to_sample,
    "hallucination/tools": phare_tools_to_sample,
    "hallucination/satirical": partial(
        phare_data_to_sample,
        module="hallucination",
        task="misinformation",
        scorer="misinformation",
    ),
    "harmful/vulnerable_misguidance": partial(
        phare_data_to_sample,
        module="harmful",
        task="vulnerable_misguidance",
        scorer="harmful_misguidance",
    ),
    "jailbreak/jailbreak": phare_jailbreak_to_sample,
}
