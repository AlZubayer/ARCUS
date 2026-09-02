"""Attribution items for the five control classes (``control_pool_v1``).

Route similarity is only interpretable if the control vectors are built exactly like the
target vectors: same objective, same graph, same attribution path. SUITE retain rows carry
no fact identity and no distractor pool, so this module gives each control item its own
pool drawn from its own category, plus a matched corrupt partner from the same category.

The five classes are kept separate everywhere. Pooling them would let a strong same-topic
effect mask the absence of a same-syntax effect, which is the confound A1 exists to rule
out.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..schema import ControlType, FactExample, Modality, normalize_answer
from ..scoring import DegenerateDistractorPool, build_distractor_set

CONTROL_POOL_VERSION = "control_pool_v1"


class ControlClass:
    """The five classes from 02_DATA_AND_SPLITS.md, never collapsed into one pool."""

    SAME_TOPIC_DIFFERENT_FACT = "same_topic_different_fact"
    SEMANTIC_NEIGHBOR = "semantic_neighbor"
    SAME_SYNTAX = "same_syntax"
    SAME_LEXICAL = "same_lexical"
    CROSS_TOPIC = "cross_topic"

    ALL = (
        SAME_TOPIC_DIFFERENT_FACT,
        SEMANTIC_NEIGHBOR,
        SAME_SYNTAX,
        SAME_LEXICAL,
        CROSS_TOPIC,
    )


@dataclass(frozen=True)
class AttributionItem:
    """One thing to attribute: a clean prompt, a matched corrupt prompt, and an outcome."""

    item_id: str
    item_class: str
    clean_question: str
    corrupt_question: str
    correct_answer: str
    distractors: tuple[str, ...]
    fact_id: str | None = None
    surface_form_id: str | None = None
    family: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def answers(self) -> tuple[str, ...]:
        return (self.correct_answer,) + self.distractors

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "item_class": self.item_class,
            "fact_id": self.fact_id,
            "surface_form_id": self.surface_form_id,
            "family": self.family,
            "clean_question": self.clean_question,
            "corrupt_question": self.corrupt_question,
            "correct_answer": self.correct_answer,
            "distractors": list(self.distractors),
            "control_pool_version": CONTROL_POOL_VERSION,
            **self.metadata,
        }


def _group_key(example: FactExample) -> tuple[str, ...]:
    """The category a retain item draws its pool and its corrupt partner from.

    Semantic rows are grouped by tier as well as topic, so a tier-0 item is never scored
    against a tier-15 pool; the tiers are different controls.
    """
    if example.control_type is ControlType.SEMANTIC:
        return (example.topic, "semantic", str(example.semantic_tier))
    return (example.topic, example.control_type.value)


def _token_len(text: str) -> int:
    return len(text.split())


def build_retain_control_items(
    retain: Sequence[FactExample],
    *,
    item_class: str,
    control_type: ControlType,
    topic: str,
    distractor_count: int,
    seed: int,
    limit: int,
    semantic_max_tier: int | None = None,
) -> tuple[list[AttributionItem], list[dict[str, Any]]]:
    """Give each retain control item its own pool and a same-category corrupt partner."""
    candidates = [
        ex
        for ex in retain
        if ex.control_type is control_type
        and ex.topic == topic
        and (
            semantic_max_tier is None
            or (ex.semantic_tier is not None and ex.semantic_tier <= semantic_max_tier)
        )
    ]

    groups: dict[tuple[str, ...], list[FactExample]] = defaultdict(list)
    for ex in candidates:
        groups[_group_key(ex)].append(ex)

    items: list[AttributionItem] = []
    refusals: list[dict[str, Any]] = []

    for key, group in sorted(groups.items()):
        ordered = sorted(group, key=lambda ex: ex.surface_form_id)
        for ex in ordered:
            if len(items) >= limit:
                break
            others = [o for o in ordered if o.surface_form_id != ex.surface_form_id]
            if not others:
                refusals.append(
                    {"item": ex.surface_form_id, "class": item_class, "reason": "no_partner"}
                )
                continue
            # Corrupt partner: same category, nearest length, deterministic tie-break.
            partner = min(
                others,
                key=lambda o: (
                    abs(_token_len(o.question) - _token_len(ex.question)),
                    o.surface_form_id,
                ),
            )
            if normalize_answer(partner.answer) == normalize_answer(ex.answer):
                refusals.append(
                    {
                        "item": ex.surface_form_id,
                        "class": item_class,
                        "reason": "partner_shares_answer",
                    }
                )
                continue
            try:
                pool = build_distractor_set(
                    ex.fact_key or _pseudo_key(ex),
                    Modality.DIRECT,
                    ex.answer,
                    [o.answer for o in others],
                    count=distractor_count,
                    seed=seed,
                )
            except DegenerateDistractorPool as exc:
                refusals.append(
                    {
                        "item": ex.surface_form_id,
                        "class": item_class,
                        "reason": "degenerate_answer_pool",
                        "detail": str(exc)[:200],
                    }
                )
                continue

            items.append(
                AttributionItem(
                    item_id=f"{item_class}|{ex.surface_form_id}",
                    item_class=item_class,
                    clean_question=ex.question,
                    corrupt_question=partner.question,
                    correct_answer=ex.answer,
                    distractors=pool.distractors,
                    surface_form_id=ex.surface_form_id,
                    family="within_control_category",
                    metadata={
                        "control_type": ex.control_type.value,
                        "semantic_tier": ex.semantic_tier,
                        "domain": ex.domain,
                        "raw_label": ex.raw_label,
                        "group": "|".join(key),
                        "linked_fact_id": (
                            ex.linked_fact_key.id if ex.linked_fact_key else None
                        ),
                    },
                )
            )
    return items, refusals


def _pseudo_key(example: FactExample):
    """Retain rows carry no fact identity; a stable pseudo-key keeps pool seeding
    deterministic without ever implying the row denotes a fact."""
    from ..schema import FactKey

    return FactKey(topic=example.topic, fact_id=f"control::{example.surface_form_id}")


def build_fact_control_items(
    pairs: Sequence[dict[str, Any]],
    *,
    item_class: str,
    exclude_fact_ids: set[str],
    family: str,
    surfaces_per_fact: int,
    limit_facts: int,
    distractor_lookup,
) -> list[AttributionItem]:
    """Control items that ARE SUITE facts, reusing their real pairs and pools."""
    by_fact: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        fact_id = f"{row['target_fact_key']['topic']}:{row['target_fact_key']['fact_id']}"
        if fact_id in exclude_fact_ids or row["family"] != family:
            continue
        by_fact[fact_id].append(row)

    items: list[AttributionItem] = []
    for fact_id in sorted(by_fact)[:limit_facts]:
        seen: set[str] = set()
        for row in sorted(by_fact[fact_id], key=lambda r: r["clean_surface_id"]):
            if row["clean_surface_id"] in seen:
                continue
            seen.add(row["clean_surface_id"])
            if len(seen) > surfaces_per_fact:
                break
            pool = distractor_lookup(row)
            if pool is None:
                continue
            items.append(
                AttributionItem(
                    item_id=f"{item_class}|{row['pair_id']}",
                    item_class=item_class,
                    clean_question=row["clean_question"],
                    corrupt_question=row["corrupt_question"],
                    correct_answer=pool.correct_answer,
                    distractors=pool.distractors,
                    fact_id=fact_id,
                    surface_form_id=row["clean_surface_id"],
                    family=row["family"],
                    metadata={"pair_id": row["pair_id"], "delta_M": row.get("delta")},
                )
            )
    return items
