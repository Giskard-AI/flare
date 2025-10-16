import logging

from pydantic import BaseModel

from flare.schema import (
    Sample,
    SampleOutputsWithScore,
    SampleWithOutputs,
    ScorerModelConfig,
    ScorerOutput,
)
from flare.scorer.base import Scorer
from flare.scorer.majority_vote_model import MajorityVoteEvaluationModel

from .prompts import EVALUATION_PROMPT, SYSTEM_PROMPT


class DebunkingResponse(BaseModel):
    reason: str
    criterion: int


class DebunkingScorer(Scorer):

    def __init__(self, models: list[ScorerModelConfig], debug: bool = False):
        super().__init__()
        self._debug = debug
        self._models = models
        self._majority_vote_model = MajorityVoteEvaluationModel(models=self._models)

    @classmethod
    def validate_sample(cls, sample: Sample) -> bool:
        valid_sample = (
            "criterion" in sample.evaluation.data
            and "context" in sample.evaluation.data
        )
        if not valid_sample:
            print(f"Invalid sample {sample.id}")
            print(f"Generations length: {len(sample.generations)}")
            print(f"Evaluation data: {str(sample.evaluation.data.keys())}")

        return valid_sample

    async def score(
        self,
        sample_with_outputs: SampleWithOutputs,
        logger: logging.Logger,
    ) -> SampleOutputsWithScore:

        model_answer = (
            sample_with_outputs.model_outputs.outputs[0].choices[0].message.content
        )
        question = sample_with_outputs.sample.generations[0].messages[-1].content

        # Set context and criterion from the evaluation data
        context = sample_with_outputs.sample.evaluation.data["context"]
        criterion = sample_with_outputs.sample.evaluation.data["criterion"]

        if self._debug:
            logger.debug(
                "--------------------------------\n"
                f"Question: {question}\n"
                f"Model answer: {model_answer}\n"
                f"Context: {context}\n"
                f"Criterion: {criterion}\n"
                "--------------------------------\n"
            )

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(
                    context=context or "", criterion=criterion or ""
                ),
            },
            {
                "role": "user",
                "content": EVALUATION_PROMPT.format(
                    question=question,
                    answer=model_answer,
                ),
            },
        ]

        logger.info(
            "Starting majority vote on sample %s", sample_with_outputs.sample.id
        )
        vote = await self._majority_vote_model.majority_vote(
            messages, decision_key="criterion", response_format=DebunkingResponse
        )
        logger.info("Majority vote done on sample %s", sample_with_outputs.sample.id)

        if self._debug:
            logger.debug(
                "--------------------------------\n"
                f"Question: {question}\n"
                f"Model answer: {model_answer}\n"
                f"Context: {context}\n"
                f"Criterion: {criterion}\n"
                f"Vote: {vote}\n"
                "--------------------------------\n"
            )

        return SampleOutputsWithScore(
            sample_with_outputs=sample_with_outputs,
            scoring=ScorerOutput(
                score=1.0 if vote.decision else 0.0,
                details={"raw_responses": vote.raw_responses},
            ),
        )
