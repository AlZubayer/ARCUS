from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class ExperimentConfig(BaseModel):
    name: str
    seed: int = 42
    output_dir: str


class ModelConfig(BaseModel):
    name: str
    revision: str | None = None
    dtype: str = "bfloat16"
    device: str = "cuda"
    trust_remote_code: bool = False


class DatasetConfig(BaseModel):
    project: str
    topic: str
    pilot_fact_limit: int | None = None
    minimum_base_accuracy: float = Field(0.8, ge=0, le=1)
    minimum_modalities_known: int = Field(2, ge=1)
    heldout_surface_fraction: float = Field(0.5, gt=0, lt=1)
    dataset_name: str | None = None
    rephrasings_dataset_name: str | None = None


class ScoringConfig(BaseModel):
    answer_metric: Literal["sequence_logprob_margin"] = "sequence_logprob_margin"
    distractor_count: int = Field(4, ge=1)
    normalize_by_answer_tokens: bool = True


class CorruptionConfig(BaseModel):
    strategy: Literal["matched_fact"] = "matched_fact"
    preserve_modality: bool = True
    preserve_topic_when_possible: bool = True
    forbid_same_fact_id: bool = True


class RouteDiscoveryConfig(BaseModel):
    method: str = "eap_ig"
    graph_granularity: str = "head_mlp_residual"
    top_k_edges: int = Field(200, ge=1)
    integrated_gradient_steps: int = Field(20, ge=2)
    discovery_split: str = "train_surface"
    evaluation_split: str = "heldout_surface"


class CircuitValidationConfig(BaseModel):
    minimality_target_fraction: float = Field(0.8, gt=0, le=1)
    necessity_target: float = Field(0.5, ge=0)
    sufficiency_target: float = Field(0.5, ge=0)
    selectivity_epsilon: float = Field(1e-6, gt=0)
    exact_intervention_required: bool = True


class RepresentationConfig(BaseModel):
    locations: list[str]
    token_policy: str
    subspace_method: str
    max_rank: int = Field(16, ge=1)
    heldout_probe_required: bool = True
    causal_projection_test: bool = True
    causal_patch_test: bool = True


class SinkConfig(BaseModel):
    enabled_after_route_validation_only: bool = True
    anchor_candidates: list[str] = ["bos", "first_position"]
    min_received_attention: float = Field(0.2, ge=0, le=1)
    require_value_contribution_measurement: bool = True
    map_all_heads: bool = True


class StatisticsConfig(BaseModel):
    bootstrap_samples: int = Field(1000, ge=100)
    permutation_samples: int = Field(1000, ge=100)
    report_per_fact: bool = True
    aggregate_by_topic: bool = True


class ModuleAConfig(BaseModel):
    experiment: ExperimentConfig
    model: ModelConfig
    dataset: DatasetConfig
    scoring: ScoringConfig
    corruption: CorruptionConfig
    route_discovery: RouteDiscoveryConfig
    circuit_validation: CircuitValidationConfig
    representation: RepresentationConfig
    sink: SinkConfig
    statistics: StatisticsConfig

    @model_validator(mode="after")
    def enforce_scientific_contract(self) -> "ModuleAConfig":
        if not self.sink.enabled_after_route_validation_only:
            raise ValueError(
                "Module A requires sink analysis to be gated behind route validation to avoid discovery bias."
            )
        if not self.circuit_validation.exact_intervention_required:
            raise ValueError("Module A requires exact causal intervention for circuit claims.")
        return self


def load_config(path: str | Path) -> ModuleAConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ModuleAConfig.model_validate(payload)
