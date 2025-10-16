from pydantic import BaseModel


class ToolRefusalResponse(BaseModel):
    correct: bool
    reason: str


class ToolStringMatchResponse(BaseModel):
    identical: bool
    reason: str


class ToolValueMatchResponse(BaseModel):
    is_close: bool
    reason: str


class APIDescription(BaseModel):
    name: str
    parameters: dict
    description: str
    generative_model: str | None = None
    language: str | None = None
    instructions: str | None = None
    knowledge_parameter: str | None = None
    original_parameters: dict | None = None


class APICall(BaseModel):
    name: str
    language: str
    original_parameters: dict
    parameters: dict | None = None
    request: str | None = None
    api_description: APIDescription
    perturbation_type: str | None = None
    generative_model: str | None = None
    knowledge_parameter_info: dict | None = None
