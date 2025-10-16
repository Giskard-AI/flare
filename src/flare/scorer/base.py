import logging
from abc import ABC, abstractmethod
from typing import Any

from flare.schema import Sample, SampleOutputsWithScore, SampleWithOutputs


class Scorer(ABC):

    @abstractmethod
    async def score(
        self,
        sample_with_outputs: SampleWithOutputs,
        logger: logging.Logger,
    ) -> SampleOutputsWithScore:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def validate_sample(cls, sample: Sample) -> bool:
        raise NotImplementedError
