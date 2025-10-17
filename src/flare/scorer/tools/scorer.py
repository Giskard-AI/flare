import builtins
import logging

import litellm
import numpy as np
from dateutil.parser import ParserError, parse

from flare.schema import (
    Sample,
    SampleOutputsWithScore,
    SampleWithOutputs,
    ScorerModelConfig,
    ScorerOutput,
)
from flare.scorer.base import Scorer
from flare.scorer.majority_vote_model import MajorityVoteEvaluationModel

from .prompts import (
    DATE_MATCH_PROMPT,
    REFUSAL_PROMPT,
    STRING_MATCH_KNOWLEDGE_PROMPT,
    STRING_MATCH_PROMPT,
    STRING_MATCH_UNIT_PROMPT,
)
from .schema import (
    APICall,
    ToolRefusalResponse,
    ToolStringMatchResponse,
    ToolValueMatchResponse,
)
from .utils import check_inclusion, extract_tool_call, robust_string_match


class ToolsScorer(Scorer):

    def __init__(self, models: list[ScorerModelConfig], debug: bool = False):
        super().__init__()
        self._debug = debug
        self._models = models
        self._majority_vote_model = MajorityVoteEvaluationModel(models=self._models)

    @classmethod
    def validate_sample(cls, sample: Sample) -> bool:
        valid_sample = (
            "api_call" in sample.evaluation.data
            and APICall.model_validate(sample.evaluation.data["api_call"])
            and "perturbation_type" in sample.evaluation.data
        )
        if not valid_sample:
            print(f"Invalid sample {sample.id}")
            print(f"Generations length: {len(sample.generations)}")
            print(f"Evaluation data: {str(sample.evaluation.data.keys())}")

        return valid_sample

    async def match_parameter_values(
        self,
        api_call: APICall,
        llm_tool_call: dict,
        param_name: str,
        param_value: str,
        logger: logging.Logger,
    ) -> tuple[bool, str]:
        # dates are tricky let's check with a dedicated parser, avoid to match "update" parameters with dates
        if (
            any(k in param_name for k in ["date", "time"])
            and "update" not in param_name
        ):
            parsed_date = False
            try:
                tool_call_date = parse(str(llm_tool_call[param_name]))
                param_value_date = parse(str(param_value))
                parsed_date = True
            except TypeError as e:
                logger.info("Type error detected")
                logger.info(param_name)
                logger.info(api_call)
                logger.info(llm_tool_call[param_name])
                logger.info(type(param_value))
                logger.info(param_value)
                raise e
            except ParserError as e:
                logger.debug("Parser error detected")
                logger.debug(param_name)
                logger.debug(llm_tool_call[param_name])
                pass

            if parsed_date and tool_call_date == param_value_date:
                return True, ""
            else:
                vote = await self._majority_vote_model.majority_vote(
                    messages=[
                        {
                            "role": "user",
                            "content": DATE_MATCH_PROMPT.format(
                                parsed_date=llm_tool_call[param_name],
                                reference_date=param_value,
                            ),
                        }
                    ],
                    decision_key="identical",
                    response_format=ToolStringMatchResponse,
                )
                if vote.decision:
                    return True, ""
                else:
                    message = f"value_mismatch - LLM evaluator found a mismatch between the parsed date and the reference date: {vote.raw_responses}"
                    return False, message

        # if the parameter is not an id but a "long" string, likely to be a text, use LLM to check if the text matches the original parameter
        # mainly to check description, messages, title, etc.
        if "id" not in param_name:
            # if the text is long enough check for parameter inclusion
            # this is not done for short text as numbers can easily be included in the parameters just by chance
            if len(str(llm_tool_call[param_name])) > 5:
                inclusion_match = check_inclusion(
                    str(llm_tool_call[param_name]), str(param_value)
                )
                if inclusion_match:
                    return True, ""

            if self._debug:
                logger.debug(
                    "Using LLM to check if the text matches the original parameter"
                )

            # if the exact string match failed, use LLM to check if the text matches the original parameter
            vote = await self._majority_vote_model.majority_vote(
                messages=[
                    {
                        "role": "user",
                        "content": STRING_MATCH_PROMPT.format(
                            value1=llm_tool_call[param_name],
                            value2=param_value,
                            parameter_name=param_name,
                        ),
                    }
                ],
                decision_key="identical",
                response_format=ToolStringMatchResponse,
            )
            if vote.decision:
                return True, ""
            else:
                message = f"value_mismatch - LLM evaluator found a mismatch between the generated text and the original parameter: {vote.raw_responses}"
                return False, message
        else:
            message = f"value_mismatch - Mismatch parameter in tool call: {param_name}, parameter in tool call: {llm_tool_call[param_name]}, parameter in original parameters: {param_value}"
            return False, message

    async def match_knowledge_parameter(
        self, api_call: APICall, llm_tool_call: dict
    ) -> tuple[bool, str]:
        knowledge_parameter = api_call.api_description.knowledge_parameter
        kp_infos = api_call.knowledge_parameter_info

        if api_call.perturbation_type == "knowledge":
            # check if the request parameter is used in the tool call which is not allowed
            if kp_infos["request_parameter"] in llm_tool_call:
                return (
                    False,
                    f"request_parameter_in_tool_call - LLM used used the request parameter {kp_infos['request_parameter']} to call the tool",
                )
            # if correct parameter is used, check if the value is correct
            if kp_infos["api_parameter"] in llm_tool_call:
                # if value is the one in the request, the model failed to do the conversion
                if robust_string_match(
                    str(llm_tool_call[kp_infos["api_parameter"]]),
                    str(kp_infos["request_parameter_value"]),
                ):
                    return (
                        False,
                        f"request_value_in_api_parameter - LLM used the request value {kp_infos['request_parameter_value']} in the tool call without conversion.",
                    )
                elif robust_string_match(
                    str(llm_tool_call[kp_infos["api_parameter"]]),
                    str(kp_infos["api_parameter_value"]),
                ):
                    return True, ""
                # in case string match did not work, use LLM to check if the value is correct
                else:
                    vote = await self._majority_vote_model.majority_vote(
                        messages=[
                            {
                                "role": "user",
                                "content": STRING_MATCH_KNOWLEDGE_PROMPT.format(
                                    text_generated=llm_tool_call[
                                        kp_infos["api_parameter"]
                                    ],
                                    text_original=kp_infos["api_parameter_value"],
                                    parameter_name=kp_infos["api_parameter"],
                                    data_source=kp_infos["data_source"],
                                ),
                            }
                        ],
                        decision_key="identical",
                        response_format=ToolStringMatchResponse,
                    )
                    if vote.decision:
                        return True, ""
                    else:
                        return False, vote.raw_responses
            else:
                return (
                    False,
                    f"missing_knowledge_parameter - LLM did not use the knowledge parameter {knowledge_parameter} to call the tool",
                )
        elif api_call.perturbation_type == "unit":
            if knowledge_parameter not in llm_tool_call:
                return (
                    False,
                    f"missing_knowledge_parameter - LLM did not use the knowledge parameter {knowledge_parameter} to call the tool",
                )
            else:
                try:
                    knowledge_parameter_type = getattr(
                        builtins,
                        api_call.api_description.parameters[knowledge_parameter],
                    )

                    # conversion can turn int to float
                    if knowledge_parameter_type in [int, float]:
                        tool_call_value = float(llm_tool_call[knowledge_parameter])
                    else:
                        tool_call_value = knowledge_parameter_type(
                            llm_tool_call[knowledge_parameter]
                        )
                except (AttributeError, ValueError):
                    tool_call_value = llm_tool_call[knowledge_parameter]
                try:
                    if (
                        kp_infos["request_parameter_value"] == tool_call_value
                        and kp_infos["request_parameter_value"] != 0
                    ):
                        return (
                            False,
                            f"value_mismatch - The model did not convert the value from the request in {kp_infos['request_unit']} into {kp_infos['api_unit']}",
                        )
                    elif np.isclose(
                        kp_infos["api_parameter_value"], tool_call_value, rtol=0.05
                    ):
                        return True, ""
                except np.exceptions.DTypePromotionError:
                    pass

                vote = await self._majority_vote_model.majority_vote(
                    messages=[
                        {
                            "role": "user",
                            "content": STRING_MATCH_UNIT_PROMPT.format(
                                value1=kp_infos["api_parameter_value"],
                                value2=tool_call_value,
                            ),
                        }
                    ],
                    decision_key="is_close",
                    response_format=ToolValueMatchResponse,
                )
                if vote.decision:
                    return True, ""
                else:
                    return False, vote.raw_responses

    async def score(
        self,
        sample_with_outputs: SampleWithOutputs,
        logger: logging.Logger,
    ) -> SampleOutputsWithScore:

        # If there is a match, the answer is CORRECT
        response = litellm.ModelResponse.model_validate(
            sample_with_outputs.model_outputs.outputs[0].raw_responses[0]
        )

        if self._debug:
            logger.debug(
                "\n".join(
                    [
                        "#" * 100,
                        f"Model: {response.model}",
                        f"Answer: {response.choices[0].message.content}",
                        f"Tool call: {response.choices[0].message.tool_calls}",
                        "#" * 100,
                    ]
                )
            )

        model_answer = (
            sample_with_outputs.model_outputs.outputs[0].choices[0].message.content
        )
        evaluation_data = sample_with_outputs.sample.evaluation.data

        api_call = APICall.model_validate(evaluation_data["api_call"])
        perturbation_type = evaluation_data["perturbation_type"]

        # the model should have declined to answer or asked for the missing parameters only if perturbation type was omission
        expected_refusal = True if perturbation_type == "omission" else False

        llm_tool_call_raw, is_tool_call_parsed_from_message = extract_tool_call(
            response
        )

        if llm_tool_call_raw is None and not is_tool_call_parsed_from_message:
            ## While considered a generation error in general, abscence of answer due to MALFORMED_FUNCTION_CALL is considered failure from the model
            # model_answer.iserror = False
            return SampleOutputsWithScore(
                sample_with_outputs=sample_with_outputs,
                scoring=ScorerOutput(
                    score=0.0,
                    details={
                        "raw_responses": f"Model answer is empty due to MALFORMED_FUNCTION_CALL",
                        "reason": "malformed_function_call",
                    },
                ),
            )
        elif llm_tool_call_raw is None:
            logger.debug(
                "Model answer is not a valid JSON object, evaluate if it was a valid refusal\n"
                f"Model answer: {model_answer}"
            )

            # if it's not expected for the model to refuse and tool call is not parsed, it's a failure
            if not expected_refusal:
                return SampleOutputsWithScore(
                    sample_with_outputs=sample_with_outputs,
                    scoring=ScorerOutput(
                        score=0.0,
                        details={
                            "raw_responses": f"JSON format is incorrect, the tool call is wrong.",
                            "reason": "json_format_incorrect",
                        },
                    ),
                )

            # if it's expected for the model to refuse but tool call is not parsed and answer is empty, it's a failure
            if model_answer.strip() == "":
                return SampleOutputsWithScore(
                    sample_with_outputs=sample_with_outputs,
                    scoring=ScorerOutput(
                        score=0.0,
                        details={
                            "raw_responses": f"Empty answer from the model.",
                            "reason": "empty_answer",
                        },
                    ),
                )

            missing_parameters = [
                k for k in api_call.original_parameters if k not in api_call.parameters
            ]
            vote = await self._majority_vote_model.majority_vote(
                messages=[
                    {
                        "role": "user",
                        "content": REFUSAL_PROMPT.format(
                            user_request=api_call.request,
                            model_answer=model_answer,
                            missing_parameters=", ".join(missing_parameters),
                        ),
                    }
                ],
                decision_key="correct",
                response_format=ToolRefusalResponse,
            )

            return SampleOutputsWithScore(
                sample_with_outputs=sample_with_outputs,
                scoring=ScorerOutput(
                    score=1.0 if vote.decision else 0.0,
                    details={
                        "raw_responses": vote.raw_responses,
                        "reason": "wrong_json_format",
                    },
                    usage=vote.usage,
                ),
            )

        if is_tool_call_parsed_from_message and litellm.supports_function_calling(
            sample_with_outputs.model_outputs.model
        ):
            logger.warning(
                "Tool call was not generated by model but was properly parsed from the assistant message."
            )

        try:
            llm_tool_call = llm_tool_call_raw["arguments"]
            if not isinstance(llm_tool_call, dict):
                logger.info(llm_tool_call)
                raise ValueError("Tool call is not a valid dictionary")

            llm_tool_name = llm_tool_call_raw["name"]
        except KeyError:
            return SampleOutputsWithScore(
                sample_with_outputs=sample_with_outputs,
                scoring=ScorerOutput(
                    score=0.0,
                    details={
                        "raw_responses": f"missing_keys - Correct JSON format but missing keys.",
                        "reason": "missing_keys",
                    },
                ),
            )

        if llm_tool_name != api_call.api_description.name:
            return SampleOutputsWithScore(
                sample_with_outputs=sample_with_outputs,
                scoring=ScorerOutput(
                    score=0.0,
                    details={
                        "raw_responses": f"wrong_tool_name - LLM used the wrong tool name: {llm_tool_name}",
                        "reason": "wrong_tool_name",
                    },
                ),
            )

        for param_name, param_type in api_call.api_description.parameters.items():
            # expected parameter value initially generated
            expected_param_value = api_call.parameters.get(param_name, None)

            if param_name == api_call.api_description.knowledge_parameter:
                valid, reason = await self.match_knowledge_parameter(
                    api_call, llm_tool_call
                )
                if not valid:
                    return SampleOutputsWithScore(
                        sample_with_outputs=sample_with_outputs,
                        scoring=ScorerOutput(
                            score=0.0,
                            details={
                                "raw_responses": reason,
                                "reason": "knowledge_parameter_mismatch",
                            },
                        ),
                    )
                llm_tool_call.pop(param_name)
                continue

            if expected_param_value is None:
                if (
                    perturbation_type == "omission"
                    and param_name in api_call.original_parameters
                ):
                    return SampleOutputsWithScore(
                        sample_with_outputs=sample_with_outputs,
                        scoring=ScorerOutput(
                            score=0.0,
                            details={
                                "raw_responses": f"Hallucinated parameter - LLM hallucinated a parameter not present in the user request",
                                "reason": "hallucinated_parameter",
                            },
                        ),
                    )
                if self._debug:
                    logger.debug(param_name)
                    logger.debug(api_call.parameters)
                    logger.debug(api_call.original_parameters)
                    logger.debug(api_call.perturbation_type)
                    logger.debug(
                        "API call not properly generated, set sample to error."
                    )
                # model_answer.iserror = True
                raise ValueError("API call not properly generated")

            # parameter missing from tool call
            if param_name not in llm_tool_call:
                return SampleOutputsWithScore(
                    sample_with_outputs=sample_with_outputs,
                    scoring=ScorerOutput(
                        score=0.0,
                        details={
                            "raw_responses": f"missing_parameter - Missing parameter in llm tool call: {param_name}",
                            "reason": "missing_parameter",
                        },
                    ),
                )

            # model found out that some parameter is missing but still perform the tool call
            elif any(
                missing_str in str(llm_tool_call[param_name]).lower()
                for missing_str in ["missing", "missing parameter"]
            ):
                return SampleOutputsWithScore(
                    sample_with_outputs=sample_with_outputs,
                    scoring=ScorerOutput(
                        score=0.0,
                        details={
                            "raw_responses": f"missing_but_call - Model figured parameters are missing but still perform the tool call",
                            "reason": "missing_but_call",
                        },
                    ),
                )

            # parameter value mismatch, call an LLM to check if the values still agree
            elif not robust_string_match(
                str(llm_tool_call[param_name]), str(expected_param_value)
            ):
                match, message = await self.match_parameter_values(
                    api_call, llm_tool_call, param_name, expected_param_value, logger
                )
                if not match:
                    return SampleOutputsWithScore(
                        sample_with_outputs=sample_with_outputs,
                        scoring=ScorerOutput(
                            score=0.0,
                            details={
                                "raw_responses": message,
                                "reason": "value_mismatch",
                            },
                        ),
                    )

            llm_tool_call.pop(param_name)
        if len(llm_tool_call) == 0:
            return SampleOutputsWithScore(
                sample_with_outputs=sample_with_outputs,
                scoring=ScorerOutput(
                    score=1.0,
                    details={
                        "raw_responses": f"correct_tool_call",
                        "reason": "correct_tool_call",
                    },
                ),
            )

        # if there are extra parameters not matched in the tool call, these are extra parameters hallucinated by the model
        return SampleOutputsWithScore(
            sample_with_outputs=sample_with_outputs,
            scoring=ScorerOutput(
                score=0.0,
                details={
                    "raw_responses": f"extra_parameters - Extra parameters in tool call: {llm_tool_call}",
                    "reason": "extra_parameters",
                },
            ),
        )
