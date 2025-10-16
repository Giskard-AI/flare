import json

import litellm
import regex


def robust_string_match(text_generated: str, text_original: str) -> bool:
    """
    Check if two texts are identical, with some robustness to slight differences in the way the texts are phrased.
    """
    return text_generated.lower().strip(". !?") == text_original.lower().strip(". !?")


def check_inclusion(text_generated: str, text_original: str) -> bool:
    """
    Check if the text_generated is included in the text_original or if the text_original is included in the text_generated.
    """
    return text_original.lower().strip(". !?") in text_generated.lower().strip(
        ". !?"
    ) or text_generated.lower().strip(". !?") in text_original.lower().strip(". !?")


def extract_tool_call(response: litellm.ModelResponse) -> tuple[dict, bool]:
    """
    Extract the tool call from the answer.
    """
    if len(response.choices) == 0:
        return None, False
    if response.choices[0].finish_reason == "MALFORMED_FUNCTION_CALL":
        return None, False
    elif response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0].model_dump()["function"]
        tool_call["arguments"] = json.loads(tool_call["arguments"])
        return tool_call, False
    else:
        pattern = regex.compile(r"\{(?:[^{}]|(?R))*\}")
        try:
            tool_call = json.loads(
                pattern.findall(response.choices[0].message.content)[0]
            )
            if "arguments" in tool_call and not isinstance(
                tool_call["arguments"], dict
            ):
                tool_call["arguments"] = json.loads(tool_call["arguments"])
            return tool_call, True
        except (json.JSONDecodeError, IndexError):
            return None, True
        except Exception as e:
            print(e)
            print(response.model_dump())
            raise e
