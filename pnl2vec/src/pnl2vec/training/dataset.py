"""Music-aware context pair generation for embedding training."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, Literal, Sequence

from pnl2vec.tokenizer.token import Token, TokenKind
from pnl2vec.tokenizer.tokenizer import AnnotatedToken, Tokenizer
from pnl2vec.tokenizer.vocabulary import Vocabulary

RelationLabel = Literal[
    "LINEAR_NEIGHBOR",
    "SAME_EVENT",
    "NEXT_EVENT",
    "PREVIOUS_EVENT",
    "SIMULTANEOUS",
    "SAME_MEASURE",
    "SAME_VOICE",
    "OTHER_HAND",
    "PHRASE_NEIGHBOR",
]


class ContextPolicy(str, Enum):
    LINEAR = "linear"
    EVENT = "event"
    MEASURE = "measure"
    VOICE = "voice"
    TEMPORAL = "temporal"
    HYBRID = "hybrid"


BOUNDARY_VALUES = frozenset(
    {
        "DOC_START",
        "DOC_END",
        "PART_START",
        "PART_END",
        "BOS",
        "EOS",
        "PAD",
        "DOC_SEP",
    }
)


@dataclass
class ContextConfig:
    policy: ContextPolicy = ContextPolicy.HYBRID
    min_window: int = 1
    max_window: int = 4
    include_same_event: bool = True
    same_event_weight: float = 0.25
    sequential_event_weight: float = 1.0
    simultaneous_event_weight: float = 1.0
    same_measure_weight: float = 0.5
    max_same_event_fraction: float = 0.35


@dataclass(frozen=True)
class ContextPair:
    center_id: int
    context_id: int
    relation: RelationLabel
    doc_id: str
    weight: float = 1.0


@dataclass
class TokenInstance:
    token_id: int
    canonical: str
    kind: TokenKind
    event_id: str | None
    measure: int | None
    voice: str | None
    hand: str | None
    offset: str | None
    event_index: int  # sequential event index within document
    token_index: int
    is_boundary: bool


def _is_boundary_token(t: Token) -> bool:
    if t.kind == TokenKind.SPECIAL:
        return str(t.value) in BOUNDARY_VALUES or str(t.value) in {
            "BOS",
            "EOS",
            "PAD",
            "DOC_SEP",
            "MEASURE_SEP",
        }
    if t.kind == TokenKind.STRUCT and str(t.value) in BOUNDARY_VALUES:
        return True
    return False


def annotated_to_instances(
    annotated: Sequence[AnnotatedToken],
    vocab: Vocabulary,
) -> list[TokenInstance]:
    instances: list[TokenInstance] = []
    event_index = -1
    last_event: str | None = object()  # type: ignore
    for i, ann in enumerate(annotated):
        t = ann.token
        if t.kind == TokenKind.SPECIAL:
            canon = f"<{t.value}>"
        else:
            canon = t.canonical()
        eid = ann.event_id
        if eid is not None and eid != last_event:
            event_index += 1
            last_event = eid
        elif eid is None and t.kind == TokenKind.EVENT:
            event_index += 1
            last_event = f"__evt_{event_index}"
            eid = last_event
        instances.append(
            TokenInstance(
                token_id=vocab.token_to_id(canon),
                canonical=canon,
                kind=t.kind,
                event_id=eid,
                measure=ann.measure,
                voice=ann.voice,
                hand=ann.hand,
                offset=ann.offset,
                event_index=event_index,
                token_index=i,
                is_boundary=_is_boundary_token(t),
            )
        )
    return instances


def generate_pairs_for_document(
    instances: Sequence[TokenInstance],
    *,
    doc_id: str,
    config: ContextConfig,
    rng: random.Random,
) -> list[ContextPair]:
    """Generate context pairs for a single document (never crosses docs)."""
    if not instances:
        return []
    policy = config.policy
    pairs: list[ContextPair] = []

    # Index by event / measure / voice / offset
    by_event: dict[str, list[int]] = defaultdict(list)
    by_measure: dict[int, list[int]] = defaultdict(list)
    by_voice: dict[str, list[int]] = defaultdict(list)
    by_offset: dict[tuple[int | None, str | None], list[int]] = defaultdict(list)
    event_order: list[str] = []
    seen_events: set[str] = set()

    for idx, inst in enumerate(instances):
        if inst.event_id is not None:
            by_event[inst.event_id].append(idx)
            if inst.event_id not in seen_events:
                seen_events.add(inst.event_id)
                event_order.append(inst.event_id)
        if inst.measure is not None:
            by_measure[inst.measure].append(idx)
        if inst.voice is not None:
            by_voice[inst.voice].append(idx)
        if inst.offset is not None or inst.measure is not None:
            by_offset[(inst.measure, inst.offset)].append(idx)

    def add_pair(i: int, j: int, relation: RelationLabel, weight: float) -> None:
        if i == j:
            return
        a, b = instances[i], instances[j]
        if a.is_boundary or b.is_boundary:
            return
        if a.token_id == vocab_pad or b.token_id == vocab_pad:
            return
        pairs.append(
            ContextPair(
                center_id=a.token_id,
                context_id=b.token_id,
                relation=relation,
                doc_id=doc_id,
                weight=weight,
            )
        )

    # vocab pad check via canonical
    vocab_pad = 0  # always reserved
    for inst in instances:
        if inst.canonical == "<PAD>":
            vocab_pad = inst.token_id
            break

    # A. Linear token window
    if policy in {ContextPolicy.LINEAR, ContextPolicy.HYBRID}:
        for i, inst in enumerate(instances):
            if inst.is_boundary:
                continue
            w = rng.randint(config.min_window, config.max_window)
            for j in range(max(0, i - w), min(len(instances), i + w + 1)):
                if j == i:
                    continue
                add_pair(i, j, "LINEAR_NEIGHBOR", 1.0)

    # B / D. Event window + sequential events
    if policy in {ContextPolicy.EVENT, ContextPolicy.VOICE, ContextPolicy.HYBRID}:
        for eidx, eid in enumerate(event_order):
            members = by_event[eid]
            # same-event
            if config.include_same_event:
                for a in members:
                    for b in members:
                        if a < b:
                            add_pair(a, b, "SAME_EVENT", config.same_event_weight)
                            add_pair(b, a, "SAME_EVENT", config.same_event_weight)
            # next/prev event
            for delta, rel in ((1, "NEXT_EVENT"), (-1, "PREVIOUS_EVENT")):
                nidx = eidx + delta
                if 0 <= nidx < len(event_order):
                    for a in members:
                        for b in by_event[event_order[nidx]]:
                            add_pair(a, b, rel, config.sequential_event_weight)  # type: ignore[arg-type]

    # C. Measure-local
    if policy in {ContextPolicy.MEASURE, ContextPolicy.HYBRID}:
        for m, members in by_measure.items():
            # sample subset to avoid explosion
            if len(members) > 40:
                members = rng.sample(members, 40)
            for a in members:
                for b in members:
                    if a != b and instances[a].event_id != instances[b].event_id:
                        add_pair(a, b, "SAME_MEASURE", config.same_measure_weight)

    # E. Temporally aligned (simultaneous)
    if policy in {ContextPolicy.TEMPORAL, ContextPolicy.HYBRID}:
        for key, members in by_offset.items():
            if key[1] is None:
                continue
            hands = {instances[i].hand for i in members}
            for a in members:
                for b in members:
                    if a >= b:
                        continue
                    if instances[a].voice != instances[b].voice or instances[a].hand != instances[b].hand:
                        rel: RelationLabel = (
                            "OTHER_HAND"
                            if instances[a].hand != instances[b].hand
                            else "SIMULTANEOUS"
                        )
                        add_pair(a, b, rel, config.simultaneous_event_weight)
                        add_pair(b, a, rel, config.simultaneous_event_weight)

    # Cap same-event fraction
    same = [p for p in pairs if p.relation == "SAME_EVENT"]
    other = [p for p in pairs if p.relation != "SAME_EVENT"]
    max_same = int(len(other) * config.max_same_event_fraction / max(1e-6, 1 - config.max_same_event_fraction))
    if len(same) > max_same and max_same >= 0:
        same = rng.sample(same, max_same) if max_same else []
    pairs = other + same
    rng.shuffle(pairs)
    return pairs


class PairStream:
    """Streaming / chunked pair iterator over documents."""

    def __init__(
        self,
        documents: Sequence,
        tokenizer: Tokenizer,
        vocabulary: Vocabulary,
        config: ContextConfig,
        *,
        seed: int = 42,
        chunk_size: int = 8192,
    ) -> None:
        self.documents = documents
        self.tokenizer = tokenizer
        self.vocabulary = vocabulary
        self.config = config
        self.seed = seed
        self.chunk_size = chunk_size

    def __iter__(self) -> Iterator[list[ContextPair]]:
        rng = random.Random(self.seed)
        buf: list[ContextPair] = []
        for doc in self.documents:
            doc_id = getattr(doc, "doc_id", "doc")
            document = getattr(doc, "document", doc)
            ann = self.tokenizer.tokenize_annotated(document)
            instances = annotated_to_instances(ann, self.vocabulary)
            pairs = generate_pairs_for_document(
                instances, doc_id=doc_id, config=self.config, rng=rng
            )
            for p in pairs:
                buf.append(p)
                if len(buf) >= self.chunk_size:
                    yield buf
                    buf = []
        if buf:
            yield buf

    def iter_pairs(self) -> Iterator[ContextPair]:
        for chunk in self:
            yield from chunk
