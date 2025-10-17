import asyncio
import json
import logging
import math
import os
import random
from typing import Any

from litellm import ModelResponse, RateLimitError, acompletion

# https://docs.litellm.ai/docs/exception_mapping


async def safe_completion(
    model_name: str,
    messages: list[dict[str, Any]],
    n: int = 1,
    nb_try: int = 1,
    ensure_json: bool = False,
    **kwargs,
) -> ModelResponse:
    nb_fail = 0
    logger = logging.getLogger(model_name)

    # If using Bedrock model, we need to set the api key to match the model region
    if "region" in kwargs:
        region = kwargs.pop("region")
        if region is not None:
            region = region.upper().replace("-", "_")
            kwargs["api_key"] = os.getenv(f"AWS_BEARER_TOKEN_BEDROCK_{region}")
    logger.info(f"Calling completion for model {model_name} with kwargs {kwargs}")
    # TODO: try with models to see how if it's working or not
    while True:
        wait_time = math.ceil(60 + 60 * random.random())
        try:
            response = await acompletion(
                model=model_name,
                messages=messages,
                n=n,
                drop_params=True,
                **kwargs,
            )
            if ensure_json:
                json.loads(response.choices[0].message.content, strict=False)
            return response
        except RateLimitError:
            # In case of RateLimit, let's not increase the retry
            # Let's stagger the retry
            logger.warning("Hitting rate limit", exc_info=True)
            logger.warning(
                "Waiting for %ss before next retry", wait_time, exc_info=True
            )
            await asyncio.sleep(wait_time)
        except Exception as e:
            nb_fail += 1
            if nb_fail >= nb_try:
                raise RuntimeError("Too many tries, cannot do completion") from e
            logger.warning("Unexpected error from calling completion", exc_info=True)
            logger.warning(
                "Waiting for %ss before next retry", wait_time, exc_info=True
            )
            await asyncio.sleep(wait_time)
