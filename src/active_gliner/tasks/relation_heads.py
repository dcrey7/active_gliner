"""Given-mention scorers; no span proposals or decoded surface-string matching."""

import json
from pathlib import Path

import torch
from gliner2.models.boundary.relations import RelationPairBatch
from torch import nn

from active_gliner.data.records import Record
from active_gliner.exact_targets import install_exact_targets


def configure(model, head: str, labels: dict[str, str], seed: int = 0) -> None:
    """Install a deterministic marker head, or select the pretrained native scorer.

    Marker parameters: (2H * H + H) + (H * R + R).
    H=768, R=17 gives 1,193,489 fully trained parameters.
    Seed zero is fixed for acquisition; training uses the run's LoRA seed.
    """
    if head not in {"classifier", "native", "marker"}:
        raise ValueError(f"Unknown relation head: {head}")
    if not labels or len(set(labels.values())) != len(labels):
        raise ValueError("Relation labels must be nonempty and unique")
    model.relation_head = head
    model.relation_labels = dict(labels)
    if head == "native":
        if getattr(model, "relation_scorer", None) is None:
            raise ValueError("The checkpoint has no native relation scorer")
        install_exact_targets(model)
    if head == "marker":
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            hidden = model.encoder.config.hidden_size
            model.marker_head = nn.Sequential(
                nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, len(labels))
            ).to(device=next(model.parameters()).device)


def mention_token_spans(mentions: list[dict], starts: list[int], ends: list[int]) -> list[tuple]:
    """Map exact character offsets to half-open word spans; never widen a mention."""
    spans = []
    for mention in mentions:
        try:
            start = starts.index(mention["start"])
            end = ends.index(mention["end"]) + 1
        except ValueError as error:
            raise ValueError(f"Mention does not align to token boundaries: {mention}") from error
        if start >= end:
            raise ValueError(f"Empty or reversed mention: {mention}")
        spans.append((start, end))
    return spans


def build_pair_batch(
    spans: list[tuple], pairs: list[tuple], n_labels: int, device
) -> RelationPairBatch:
    """Return pair-major, label-minor coordinates, with no thresholds or caps."""
    if n_labels <= 0:
        raise ValueError("At least one relation label is required")
    if any(start < 0 or end <= start for start, end in spans):
        raise ValueError("Token spans must be nonempty and half-open")
    rows = []
    for head, tail in pairs:
        if head == tail or not (0 <= head < len(spans) and 0 <= tail < len(spans)):
            raise ValueError(f"Invalid ordered mention pair: {(head, tail)}")
        rows.extend((r, *spans[head], *spans[tail]) for r in range(n_labels))
    indices = torch.tensor(rows, dtype=torch.long, device=device).reshape(-1, 5)
    count = len(rows)
    return RelationPairBatch(
        batch_index=torch.zeros(count, dtype=torch.long, device=device),
        relation_index=indices[:, 0],
        head_start=indices[:, 1],
        head_end=indices[:, 2],
        tail_start=indices[:, 3],
        tail_end=indices[:, 4],
        head_prob=torch.ones(count, device=device),
        tail_prob=torch.ones(count, device=device),
        pair_mask=torch.ones(count, dtype=torch.bool, device=device),
    )


def marker_text_positions(record: Record, head: int, tail: int) -> tuple[str, tuple[int, int]]:
    """Insert the existing typed markers and retain their exact character positions."""
    events = {}
    for index, role in ((head, "H"), (tail, "T")):
        mention = record.gold["entities"][index]
        if not 0 <= mention["start"] < mention["end"] <= len(record.text):
            raise ValueError(f"Invalid mention offsets: {mention}")
        events.setdefault(mention["start"], []).append((1, f"[{role}:{mention['label']}] ", role))
        events.setdefault(mention["end"], []).append((0, f" [/{role}]", None))
    parts, positions, offset, length = [], {}, 0, 0
    for position, markers in sorted(events.items()):
        text = record.text[offset:position]
        parts.append(text)
        length += len(text)
        for _, text, role in sorted(markers):
            if role is not None:
                positions[role] = length
            parts.append(text)
            length += len(text)
        offset = position
    parts.append(record.text[offset:])
    return "".join(parts), (positions["H"], positions["T"])


def marker_token_positions(offsets: list, positions: tuple[int, int]) -> tuple[int, int]:
    result = []
    for position in positions:
        matches = [i for i, (start, end) in enumerate(offsets) if start <= position < end]
        if len(matches) != 1:
            raise ValueError(f"Marker at {position} does not map to exactly one subword")
        result.append(matches[0])
    return tuple(result)


def pool_markers(hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Concatenate head then tail states, preserving pair direction."""
    if positions.shape != (hidden.shape[0], 2):
        raise ValueError("Expected two marker positions per input")
    if torch.any(positions < 0) or torch.any(positions >= hidden.shape[1]):
        raise ValueError("Marker position outside the encoded input")
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    return torch.cat((hidden[rows, positions[:, 0]], hidden[rows, positions[:, 1]]), dim=-1)


def pair_logits(model, rows: list[tuple[Record, int, int]], labels: dict) -> torch.Tensor:
    """Score supplied pairs against every label; retain gradients during training."""
    if dict(labels) != model.relation_labels:
        raise ValueError("Relation labels differ from the saved head schema")
    device = next(model.parameters()).device
    if model.relation_head == "marker":
        marked = [marker_text_positions(record, i, j) for record, i, j in rows]
        batch = model.processor.tokenizer(
            [text for text, _ in marked],
            padding=True,
            truncation=False,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = batch.pop("offset_mapping").tolist()
        positions = torch.tensor(
            [
                marker_token_positions(offset, positions)
                for offset, (_, positions) in zip(offsets, marked, strict=True)
            ],
            device=device,
        )
        hidden = model.encoder(**batch.to(device)).last_hidden_state
        pooled = pool_markers(hidden, positions)
        logits = model.marker_head(pooled.to(next(model.marker_head.parameters()).dtype))
        order = [list(model.relation_labels).index(label) for label in labels]
        return logits[:, order]
    if model.relation_head != "native":
        raise ValueError("pair_logits requires native or marker")
    # Group pairs of the same sentence to share the encoder pass.
    groups = {}
    for position, (record, i, j) in enumerate(rows):
        groups.setdefault(id(record), []).append((position, record, i, j))
    output = [None] * len(rows)
    schema = model.create_schema().relations(list(labels.values())).build()
    processor = model.processor
    was_training = processor.is_training
    processor.change_mode(is_training=False)
    try:
        for group in groups.values():
            record = group[0][1]
            batch = processor.collate_fn_inference([(record.text, schema)])
            spans = mention_token_spans(
                record.gold["entities"], batch.start_mappings[0], batch.end_mappings[0]
            )
            core = model._encode_core(batch)
            if core["word_offsets"][0] != 0:
                raise ValueError("Unexpected text prefix in native relation input")
            specs = {entry["relation_type"]: entry for entry in core["rel_specs"][0]}
            if set(specs) != set(labels.values()):
                raise ValueError("Native schema lost relation queries")
            states = torch.stack([specs[label]["query_state"] for label in labels.values()])[None]
            pairs = build_pair_batch(spans, [(i, j) for _, _, i, j in group], len(labels), device)
            logits = model.relation_scorer(core["text_states"], states, None, pairs)
            for item, logit in zip(group, logits.reshape(-1, len(labels)), strict=True):
                output[item[0]] = logit
    finally:
        processor.change_mode(is_training=was_training)
    return torch.stack(output)


def save_metadata(model, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "relation_head.json").write_text(
        json.dumps(
            {
                "head": model.relation_head,
                "labels": model.relation_labels,
                "marker_parameters": (
                    2 * model.hidden_size**2
                    + model.hidden_size
                    + model.hidden_size * len(model.relation_labels)
                    + len(model.relation_labels)
                )
                if model.relation_head == "marker"
                else 0,
            },
            indent=2,
        )
        + "\n"
    )
