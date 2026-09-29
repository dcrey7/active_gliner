"""Round 32: given-mention relation heads (native scorer and marker pooling)."""

import pytest
import torch

from active_gliner.data.records import Record
from active_gliner.tasks import relation_heads


def _record():
    return Record(
        id="crossre:en-US:x",
        doc_id=None,
        text="Marie Curie worked in Paris .",
        task="relations",
        locale="en-US",
        source_split="train",
        gold={
            "entities": [
                {"start": 0, "end": 11, "text": "Marie Curie", "label": "person"},
                {"start": 22, "end": 27, "text": "Paris", "label": "location"},
            ],
            "relations": [],
        },
    )


# Word boundaries of "Marie Curie worked in Paris ."
STARTS = [0, 6, 12, 19, 22, 28]
ENDS = [5, 11, 18, 21, 27, 29]


def test_mentions_map_to_half_open_word_spans():
    spans = relation_heads.mention_token_spans(_record().gold["entities"], STARTS, ENDS)
    assert spans == [(0, 2), (4, 5)]


def test_mention_inside_a_word_is_rejected():
    mention = {"start": 1, "end": 5, "text": "arie", "label": "person"}
    with pytest.raises(ValueError, match="token boundaries"):
        relation_heads.mention_token_spans([mention], STARTS, ENDS)


def test_pair_batch_is_pair_major_label_minor():
    batch = relation_heads.build_pair_batch([(0, 2), (4, 5)], [(0, 1), (1, 0)], 3, "cpu")
    assert batch.relation_index.tolist() == [0, 1, 2, 0, 1, 2]
    assert batch.head_start.tolist() == [0, 0, 0, 4, 4, 4]
    assert batch.tail_start.tolist() == [4, 4, 4, 0, 0, 0]
    assert batch.pair_mask.all()


@pytest.mark.parametrize("pairs", [[(0, 0)], [(0, 2)], [(-1, 0)]])
def test_pair_batch_rejects_bad_pairs(pairs):
    with pytest.raises(ValueError):
        relation_heads.build_pair_batch([(0, 2), (4, 5)], pairs, 3, "cpu")


def test_marker_positions_point_at_the_opening_markers():
    text, (head, tail) = relation_heads.marker_text_positions(_record(), 0, 1)
    assert text.startswith("[H:person] Marie Curie [/H]")
    assert text[head:].startswith("[H:person]")
    assert text[tail:].startswith("[T:location]")


def test_marker_positions_follow_the_requested_direction():
    text, (head, tail) = relation_heads.marker_text_positions(_record(), 1, 0)
    assert text[head:].startswith("[H:location]")
    assert text[tail:].startswith("[T:person]")


def test_marker_token_positions_need_exactly_one_subword():
    offsets = [(0, 0), (0, 3), (3, 8), (8, 12)]
    assert relation_heads.marker_token_positions(offsets, (0, 8)) == (1, 3)
    with pytest.raises(ValueError):
        relation_heads.marker_token_positions(offsets, (20, 0))


def test_pool_markers_keeps_head_then_tail():
    hidden = torch.arange(2 * 4 * 2, dtype=torch.float).reshape(2, 4, 2)
    pooled = relation_heads.pool_markers(hidden, torch.tensor([[1, 3], [2, 0]]))
    assert pooled.tolist() == [[2.0, 3.0, 6.0, 7.0], [12.0, 13.0, 8.0, 9.0]]


def test_pool_markers_rejects_positions_outside_the_input():
    hidden = torch.zeros(1, 4, 2)
    with pytest.raises(ValueError):
        relation_heads.pool_markers(hidden, torch.tensor([[0, 4]]))
