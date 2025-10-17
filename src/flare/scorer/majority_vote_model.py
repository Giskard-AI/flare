import asyncio
import json
import logging
from typing import Type

import regex
from litellm import ModelResponse
from pydantic import BaseModel

from flare.complete import safe_completion
from flare.schema import FlareModel, OutputUsage, ScorerModelConfig


def extract_json_object(response: str) -> dict:
    """
    Extract the json object from the response.
    """
    pattern = regex.compile(r"\{(?:[^{}]|(?R))*\}")
    try:
        return json.loads(pattern.findall(response)[0])
    except (json.JSONDecodeError, IndexError) as e:
        return None


class VoteException(Exception):
    def __init__(self, message):
        super().__init__(message)


class MajorityVote(FlareModel):
    decision: bool
    raw_responses: dict[str, dict]
    usage: dict[str, OutputUsage]


class MajorityVoteEvaluationModel(BaseModel):
    models: list[ScorerModelConfig]

    async def majority_vote(
        self,
        messages: list[dict],
        decision_key: str,
        response_format: Type[BaseModel],
    ) -> MajorityVote:

        votes = {}

        responses = await asyncio.gather(
            *[
                safe_completion(
                    model.litellm_model,
                    messages,
                    response_format=response_format,
                    **model.generation_kwargs.model_dump(),
                )
                for model in self.models
            ]
        )

        evaluation_usage = {}
        for model, response in zip(self.models, responses):
            try:
                response_json = extract_json_object(response.choices[0].message.content)
                votes[model.litellm_model] = {
                    "response": response_json[decision_key],
                    "weight": model.weight,
                }
                if "reason" in response_json:
                    votes[model.litellm_model]["reason"] = response_json["reason"]
                
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
                evaluation_usage[model.litellm_model] = response_usage

            except Exception as e:
                logging.error(
                    "Error in evaluation with model",
                    model.litellm_model,
                    ":",
                    repr(response),
                )
                continue

        total_weight = sum([model.weight for model in self.models])

        pass_weight_sum = sum(
            vote["weight"] for vote in votes.values() if vote["response"] == 1
        )
        fail_weight_sum = sum(
            vote["weight"] for vote in votes.values() if vote["response"] == 0
        )

        # Check for consensus
        if pass_weight_sum > total_weight / 2:
            return MajorityVote(decision=True, raw_responses=votes, usage=evaluation_usage)
        elif fail_weight_sum > total_weight / 2:
            return MajorityVote(decision=False, raw_responses=votes, usage=evaluation_usage)
        else:
            raise VoteException("No consensus reached")
