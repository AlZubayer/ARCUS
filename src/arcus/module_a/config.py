from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

CONFIG_VERSION = "module_a_config_v2"

#: System prompt used by the upstream SUITE evaluation for Llama-3.2-3B-Instruct
#: (configs/model/Llama-3.2-3B-Instruct.yaml in AmitPeleg/forget-narrowly-retain-broadly).
#: Reproduced so ARCUS scores this model the way the benchmark's authors do.
SUITE_SYSTEM_PROMPT = (
    "You are a helpful assistant. Respond with only the final answer to the question. "
    "Your output must contain only the answer itself."
)


class ExperimentConfig(BaseModel):
    name: str
    seed: int = 42
    output_dir: str
    artifact_dir: str = "artifacts"


class ModelConfig(BaseModel):
    name: str
    revision: str
    tokenizer_name: str | None = None
    tokenizer_revision: str | None = None
    dtype: Literal["float32", "bfloat16", "float16"] = "float32"
    device: str = "cuda"
    attn_implementation: Literal["eager", "sdpa"] = "eager"
    trust_remote_code: bool = False

    @model_validator(mode="after")
    def require_pinned_revision(self) -> "ModelConfig":
        if not self.revision or self.revision.lower() in {"null", "none", "main"}:
            raise ValueError(
                "model.revision must pin an exact commit sha. A floating revision cannot "
                "appear in a reproducible run manifest."
            )
        return self

    @property
    def resolved_tokenizer_name(self) -> str:
        return self.tokenizer_name or self.name

    @property
    def resolved_tokenizer_revision(self) -> str:
        return self.tokenizer_revision or self.revision


class PromptConfig(BaseModel):
    """Registered prompt policy. Changing any field is a new scoring lineage."""

    template: Literal["chat", "raw_completion"] = "chat"
    template_version: str = "llama3_chat_suite_v1"
    system_prompt: str = SUITE_SYSTEM_PROMPT
    add_generation_prompt: bool = True
    # The Llama-3 chat template already emits <|begin_of_text|>; adding specials again
    # would produce a double BOS and silently shift every token position.
    add_special_tokens: bool = False
    raw_completion_template: str = "Question: {question}\nAnswer:"
    answer_prefix: str = " "


class DatasetConfig(BaseModel):
    project: str = "forget-narrowly-retain-broadly"
    topic: str
    dataset_id: str
    dataset_revision: str
    rephrasings_dataset_id: str
    rephrasings_revision: str
    pilot_fact_limit: int | None = None
    minimum_base_accuracy: float = Field(0.8, ge=0, le=1)
    minimum_modalities_known: int = Field(2, ge=1)
    #: Splits whose surfaces are scored when screening base knowledge. Stress (indirect)
    #: is included because it is the only other fact-discriminative modality; A0 screening
    #: is not candidate selection, and A1 still reads the discovery split alone.
    known_fact_splits: list[Literal["validation", "discovery", "stress"]] = [
        "validation",
        "stress",
    ]
    #: Modalities eligible to carry the factual margin. ``null`` auto-detects, excluding
    #: any cell whose distractor pool is degenerate under the configured policy.
    known_fact_modalities: list[str] | None = None
    topic_sweep_on_shortfall: bool = True
    minimum_pilot_facts: int = Field(5, ge=1)

    @model_validator(mode="after")
    def require_pinned_revisions(self) -> "DatasetConfig":
        for field_name in ("dataset_revision", "rephrasings_revision"):
            value = getattr(self, field_name)
            if not value or value.lower() in {"null", "none", "main"}:
                raise ValueError(
                    f"dataset.{field_name} must pin an exact revision sha (see the dataset audit)."
                )
        return self


class ScoringConfig(BaseModel):
    answer_metric: Literal["sequence_logprob_margin"] = "sequence_logprob_margin"
    answer_score: Literal["mean_logprob", "sum_logprob"] = "mean_logprob"
    scoring_version: str = "answer_score_mean_logprob_v1"
    margin_version: str = "factual_margin_v1"
    correctness_version: str = "correct_over_distractors_v1"
    distractor_count: int = Field(4, ge=1)
    distractor_policy: str = "same_topic_same_modality_v1"
    #: Clean/corrupt pairs whose |delta| falls below this are scientifically unusable:
    #: every normalized causal metric divides by that delta (03_EXPERIMENT_PROTOCOL.md section 1).
    min_abs_delta: float = Field(0.5, gt=0)
    batch_size: int = Field(8, ge=1)


class CorruptionConfig(BaseModel):
    strategy: Literal["matched_fact"] = "matched_fact"
    corruption_policy_version: str = "pair_families_v1"
    families: list[str] = [
        "same_topic_fact_swap",
        "semantic_neighbor",
        "same_syntax",
        "same_lexical_different_meaning",
        "cross_topic_matched",
        "random_token_control",
    ]
    preserve_modality: bool = True
    preserve_topic_when_possible: bool = True
    preserve_surface_kind: bool = True
    forbid_same_fact_id: bool = True
    max_pairs_per_fact_family: int = Field(4, ge=1)
    #: Clean surfaces per eligible fact. Kept small so the pilot stays auditable.
    clean_surfaces_per_fact: int = Field(2, ge=1)
    #: The modality that can carry the margin; reverse is degenerate (see the audit).
    pair_modality: Literal["direct", "indirect"] = "direct"
    pair_clean_split: Literal["validation", "discovery", "stress"] = "validation"


class PatchingConfig(BaseModel):
    """Exact-intervention scope. Budgets are explicit so a scan is never accidental."""

    hook_map_version: str = "llama_hook_map_v1"
    components: list[str] = ["resid_pre", "attn_out", "mlp_out"]
    token_policies: list[str] = ["last_prompt_token"]
    scan_components: list[str] = ["resid_pre"]
    scan_token_policy: str = "final_k_prompt_tokens"
    scan_final_k: int = Field(8, ge=1)
    layer_stride: int = Field(1, ge=1)
    directions: list[Literal["clean_to_corrupt", "corrupt_to_clean"]] = [
        "clean_to_corrupt",
        "corrupt_to_clean",
    ]
    max_pairs_scanned: int = Field(4, ge=1)
    #: Capture-only hooks clone tensors, so any nonzero drift is a bug, not noise.
    parity_tolerance: float = Field(0.0, ge=0)
    self_patch_tolerance: float = Field(0.0, ge=0)


class RouteDiscoveryConfig(BaseModel):
    method: str = "eap_ig"
    graph_granularity: str = "head_mlp_residual"
    top_k_edges: int = Field(200, ge=1)
    integrated_gradient_steps: int = Field(20, ge=2)
    discovery_split: str = "discovery"
    evaluation_split: str = "validation"


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
    prompt: PromptConfig = PromptConfig()
    dataset: DatasetConfig
    scoring: ScoringConfig
    corruption: CorruptionConfig
    patching: PatchingConfig = PatchingConfig()
    route_discovery: RouteDiscoveryConfig
    circuit_validation: CircuitValidationConfig
    representation: RepresentationConfig
    sink: SinkConfig
    statistics: StatisticsConfig
    config_version: str = CONFIG_VERSION

    @model_validator(mode="after")
    def enforce_scientific_contract(self) -> "ModuleAConfig":
        if not self.sink.enabled_after_route_validation_only:
            raise ValueError(
                "Module A requires sink analysis to be gated behind route validation to avoid "
                "discovery bias."
            )
        if not self.circuit_validation.exact_intervention_required:
            raise ValueError("Module A requires exact causal intervention for circuit claims.")
        return self


def config_sha256(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def load_config(path: str | Path) -> ModuleAConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ModuleAConfig.model_validate(payload)
