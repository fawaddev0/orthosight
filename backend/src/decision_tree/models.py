"""Validated, read-only models for the clinical decision tree data."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


class TreeModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class RedFlag(TreeModel):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    next_node_id: str = Field(alias="nextNodeId", min_length=1)


class Option(TreeModel):
    value: str = Field(min_length=1)
    label: str = Field(min_length=1)
    next_node_id: str = Field(alias="nextNodeId", min_length=1)


class ScreeningNode(TreeModel):
    id: str = Field(min_length=1)
    type: Literal["screening"]
    title: str = Field(min_length=1)
    red_flags: tuple[RedFlag, ...] = Field(alias="redFlags")
    default_next_node_id: str = Field(alias="defaultNextNodeId", min_length=1)


class QuestionNode(TreeModel):
    id: str = Field(min_length=1)
    type: Literal["question"]
    title: str = Field(min_length=1)
    question_text: str = Field(alias="questionText", min_length=1)
    patient_text: str | None = Field(default=None, alias="patientText")
    options: tuple[Option, ...] = Field(min_length=1)


class QuestionnaireRule(TreeModel):
    condition: str = Field(min_length=1)
    label: str | None = None
    next_node_id: str = Field(alias="nextNodeId", min_length=1)


class QuestionnaireEvalNode(TreeModel):
    id: str = Field(min_length=1)
    type: Literal["questionnaire_eval"]
    title: str = Field(min_length=1)
    questionnaire_id: str | None = Field(
        default=None,
        alias="questionnaireId",
        validation_alias=AliasChoices("questionnaireId", "questionnaire_id"),
        min_length=1,
    )
    rules: tuple[QuestionnaireRule, ...] | None = Field(default=None, min_length=1)
    question_text: str | None = Field(default=None, alias="questionText", min_length=1)
    options: tuple[Option, ...] | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_questionnaire_shape(self) -> QuestionnaireEvalNode:
        rule_shape = self.questionnaire_id is not None and self.rules is not None
        option_shape = self.question_text is not None and self.options is not None
        if not rule_shape and not option_shape:
            raise ValueError("questionnaire_eval requires questionnaire rules or question options")
        return self


class DiagnosisNode(TreeModel):
    id: str = Field(min_length=1)
    type: Literal["diagnosis"]
    title: str = Field(min_length=1)
    primary_diagnosis: str = Field(alias="primaryDiagnosis", min_length=1)
    differentials: tuple[str, ...]
    confidence_score: int = Field(alias="confidenceScore", ge=0, le=100)
    icd_code: str | None = Field(default=None, alias="icdCode")


Node = Annotated[
    ScreeningNode | QuestionNode | QuestionnaireEvalNode | DiagnosisNode,
    Field(discriminator="type"),
]


class Layer(TreeModel):
    id: str = Field(pattern=r"^[1-4]$")
    version: str = Field(min_length=1)
    root_node_id: str = Field(alias="rootNodeId", min_length=1)
    nodes: dict[str, Node]
