# from .factuality.scorer import FactualityScorer
from flare.schema import ModelConfig, ScorerModelConfig
from flare.scorer.base import Scorer
from flare.scorer.bias.scorer import BiasesScorer
from flare.scorer.debunking.scorer import DebunkingScorer
from flare.scorer.factuality.scorer import FactualityScorer
from flare.scorer.jailbreak.scorer import (
    EncodingJailbreakScorer,
    FramingJailbreakScorer,
    InjectionJailbreakScorer,
)
from flare.scorer.misinformation.scorer import MisinformationScorer
from flare.scorer.tools.scorer import ToolsScorer
from flare.scorer.vulnerable_misguidance.scorer import HarmfulMisguidanceScorer

SCORERS = {
    "factuality": FactualityScorer,
    "debunking": DebunkingScorer,
    "tools": ToolsScorer,
    "misinformation": MisinformationScorer,
    "biases/story_generation": BiasesScorer,
    "harmful_misguidance": HarmfulMisguidanceScorer,
    "jailbreak/encoding": EncodingJailbreakScorer,
    "jailbreak/injection": InjectionJailbreakScorer,
    "jailbreak/framing": FramingJailbreakScorer,
}


def get_scorer(
    scorer_name: str, models: list[ScorerModelConfig], generators: list[ModelConfig], *args, **kwargs
) -> Scorer:
    if scorer_name == "biases/story_generation":
        kwargs["generators"] = generators
    return SCORERS[scorer_name](models, *args, **kwargs)
