# Flare


## Install
- `make setup`

or 

- `uv sync`

## Needed in env variables:
- openai key:  export OPENAI_API_KEY=...
- mistral: export MISTRAL_API_KEY=....
- gemini : export GEMINI_API_KEY=...
- open router : export OPENROUTER_API_KEY=...

## Example run:

`uv run python ./src/flare/main.py  --config-path ./configs/example_config.json --sample-path ./samples/ --name simple-test`
![Example run](./docs/example-run.png)

## Data structure

When launching a run, the following folder and files will be created :
- `<run_name>/generation/<model_name>/<sample_id.json>`

Inside this JSON, two additional keys are going to be added compared to the sample : 

- `generation_metadata` : object containing model config, run_name, nb_try and so on
- `generation`: object with as id the question id, and as values the list of generated answers

```
    "generation_metadata": {
        "model": {
            "name": "GPT 4o",
            "litellm_model": "openai/gpt-4o",
            "parallelism": 8
        },
        "nb_try": 5,
        "run_name": "simple-test"
    },
        "generation": {
        "ed64f99b-1ffe-463d-9129-36b7e1623f1e": [
            {
                "raw_answer": {
                 /* raw answer from llm call*/
                },
                "answer": "<llm story>"
            },
            ...
            ]
        }

```

- `<run_name>/result/<model_name>/<task_name>/<sample_id.json>`

Depends on the scorer, for bias, contains a 'result' object in the sample


Notes: samples from bias dataset should be usable as is

## Create and register a scorer

To create a new scorer in Flare : 
- create a new module inside of `flare.scorer`
- add the configuration for the scorer inside of the JSON config 'scorers.<name>' with an object
- have a class extending `flare.scorer.base.Scorer`
  - implement `validate_sample` to ensure the sample got everything you need as params to score (not the generation part)
  - implement `score` with the scoring method
  - the `__init__` will receive (and must handle) the provided configuration as kwargs
- register the new scorer in flare.scorer.get_scorer

And that's it ! Now it should all setup !
