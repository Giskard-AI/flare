from typing import Literal

from pydantic import BaseModel, Field


class DemographicAttributes(BaseModel):
    age: Literal[
        "child (0-12)",
        "adolescent (13-17)",
        "young adult (18-29)",
        "adult (30-44)",
        "middle-aged (45-64)",
        "senior (65+)",
        "unknown",
    ]
    gender: Literal["man", "woman", "non-binary", "other", "unknown"]
    gender_alignment: Literal["cisgender", "transgender", "other", "unknown"]
    sexual_orientation: Literal[
        "heterosexual",
        "homosexual",
        "bisexual",
        "pansexual",
        "asexual",
        "other",
        "unknown",
    ]
    geographic_origin: Literal[
        "north america (e.g. United States, Canada, etc)",
        "south or central america (e.g. Mexico, Brazil, Argentina, Cuba, etc)",
        "europe (e.g. United Kingdom, Spain, Russia, Greece, etc)",
        "south eastern asia (e.g. Thailand, Vietnam, Philippines, Malaysia, Indonesia, etc)",
        "eastern asia (e.g. Japan, Korea, China, etc)",
        "central asia (e.g. Kazakhstan, Kyrgyzstan, Tajikistan, Turkmenistan, Uzbekistan, etc)",
        "southern asia (e.g. India, Pakistan, Sri Lanka, Nepal, etc)",
        "northern africa (e.g. Egypt, Sudan, Algeria, Morocco, Tunisia, etc)",
        "sub-saharan africa (e.g. Nigeria, Ethiopia, Kenya, Tanzania, Uganda, etc)",
        "middle east (e.g. Saudi Arabia, Iran, Afghanistan, etc)",
        "oceania (e.g. Australia, New Zealand, Fiji, etc)",
        "other",
        "unknown",
    ]
    education: Literal[
        "basic (e.g. less than high school, or high school)",
        "undergraduate (e.g. bachelor's degree, some college)",
        "graduate (e.g. master's degree, doctoral degree)",
        "unknown",
    ]
    religion: Literal[
        "Christian",
        "Jewish",
        "Muslim",
        "Hindu",
        "Buddhist",
        "Atheist/Agnostic",
        "other",
        "unknown",
    ]
    political_orientation: Literal[
        "conservative",
        "centrist",
        "progressive",
        "unknown",
    ]
    health_status: Literal["good", "fair", "poor", "unknown"]
    disability_status: Literal[
        "no disability",
        "some disability",
        "unknown",
    ]
    literacy_status: Literal["literate", "illiterate", "unknown"]
    immigration_status: Literal[
        "citizen (native-born or naturalized)",
        "immigrant",
        "unknown",
    ]
    urbanicity: Literal["urban", "rural", "unknown"]
    marital_status: Literal[
        "single",
        "married",
        "divorced or separated",
        "widowed",
        "domestic partnership",
        "unknown",
    ]
    parental_status: Literal[
        "childless",
        "with children",
        "unknown",
    ]
    income_level: Literal["low income", "middle income", "high income", "unknown"]
    employment_status: Literal[
        "stable employment",
        "precarious employment",
        "unemployed",
        "student",
        "retired",
        "other",
        "unknown",
    ]
    professional_field: Literal[
        "science, technology, engineering, and math (STEM)",
        "healthcare and medicine",
        "education and training",
        "media and communication",
        "arts and creative industries",
        "business, finance, legal",
        "administrative assistance and support services",
        "social services",
        "public administration",
        "trades and manual labor",
        "agriculture",
        "other",
        "unknown",
    ]
    housing_status: Literal[
        "homeowner",
        "renter",
        "homeless",
        "other",
        "unknown",
    ]


class AttributeExtractionResponse(BaseModel):
    evidences: str = Field(
        description="A summary explaining your choices with evidence from the story"
    )
    attributes: DemographicAttributes = Field(
        description="The demographic attributes of the main character"
    )
