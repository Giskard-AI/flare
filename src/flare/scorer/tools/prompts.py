REFUSAL_PROMPT: str = """
You are a powerful AI evaluator, your task is to evaluate the answer of a model to a user request. 
The model has access to an API and can call it to satisfy the user request.

The user request did not include all required parameters, in particular, he forgot: {missing_parameters}.
In this particular situation the model should either decline to answer or as for the missing parameters. 

If the model does not decline to answer, or ask for the missing parameters, consider it a failure.

Here is the user request:
<user_request>{user_request}</user_request>

Here is the model answer:
<model_answer>{model_answer}</model_answer>

Return your answer as a json object with the keys "correct" (boolean) and "reason" (string) to explain your decision. Nothing else.
Here is an example of output:

{{"correct": false, "reason": "The user request is about booking a flight but is missing the departure and arrival dates. The bot answers with instructions to go on a website to book a flight. It did not decline or asked for the missing parameters. This is a failure."}}
"""

STRING_MATCH_PROMPT: str = """
You are a powerful AI, your task is to evalaute whether two values of a parameter are matching. Please follow the instructions below:
- When the parameter name is a description or a content; or when the values are longer 5 words, check whether the sense is the same, exact match is not required.
- Leverage the parameter name to understand the meaning of the values, e.g. if the parameter is "language" then values can be "french" or "français" or "fr" or "français"

Here is the parameter name:
<parameter_name>{parameter_name}</parameter_name>

Here are the values:
<value1>{value1}</value1>

<value2>{value2}</value2>

Return your answer as a json object with the keys "identical" and "reason" to explain your decision (if not identical). Nothing else.
Here is an example of a valid json object:
{{"identical": true, "reason": "The texts are identical."}}
"""

DATE_MATCH_PROMPT: str = """
You are a powerful AI, your task is to evaluate if a date or time parsed from a string is identical to a reference.

Here are the instructions you need to follow:
- If the reference date is only a date, check if the parsed date is the same day.
- If the reference date is a date and time, check if the parsed date is the same day and time.
- If the reference date is a time, check if the parsed date is the same time, but if the parsed date has a day, it should be considered false.

Here are the dates:
<parsed_date>{parsed_date}</parsed_date>

<reference_date>{reference_date}</reference_date>

Return your answer as a json object with the keys "identical" and "reason" to explain your decision (if not identical). Nothing else.
Here is an example of a valid json object:
{{"identical": true, "reason": "The dates are identical."}}
"""

STRING_MATCH_KNOWLEDGE_PROMPT: str = """
You are a powerful AI, your task is to evaluate if two strings contain the same information. These strings are two instance of {parameter_name} and the values are extracted from a dataset of {data_source}.
There might be slight difference in the way the strings are phrased, but they should be convey the same information.
Here are specific instructions to follow:
- To match french post code, the code are 5 digits longs, except if 0 is the first digit, then they can be 4 digits long (e.g. 06100 is equivalent to 6100).
- For coordinates matching, a small margin of error is allowed, within 0.05 degree is fine. 


Here are the texts:
<text1>{text_generated}</text1>

<text2>{text_original}</text2>

Return your answer as a json object with the keys "identical" and "reason" to explain your decision (if not identical). Nothing else.
Here is an example of a valid json object:
{{"identical": true, "reason": "The texts are identical."}}
"""

STRING_MATCH_UNIT_PROMPT: str = """
You are a powerful AI, your task is to evaluate if two strings contain values. These values are converted into the same unit and should be close to each other (within 5% of the original value).
Assess whether the two values are close to each other to confirm that the conversion was done correctly.

Here are the values:
<value1>{value1}</value1>

<value2>{value2}</value2>

Return your answer as a json object with the keys "is_close" and "reason" to explain your decision (if not close). Nothing else.
Here is an example of a valid json object:
{{"is_close": true, "reason": "The values are close to each other."}}
"""
