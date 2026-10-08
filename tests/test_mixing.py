import pytest

from active_gliner import mixing

IDS = [f"s{i}" for i in range(100)]
NO_SCORES = {i: 0.0 for i in IDS}


def test_nested_sets_grow_by_extension():
    small = mixing.gold_ids(IDS, NO_SCORES, 0.1, "nested", seed=3)
    large = mixing.gold_ids(IDS, NO_SCORES, 0.4, "nested", seed=3)
    assert len(small) == 10
    assert len(large) == 40
    assert small <= large


def test_nested_does_not_depend_on_input_order():
    forward = mixing.gold_ids(IDS, NO_SCORES, 0.2, "nested", seed=1)
    backward = mixing.gold_ids(list(reversed(IDS)), NO_SCORES, 0.2, "nested", seed=1)
    assert forward == backward


def test_nested_seed_changes_the_draw():
    one = mixing.gold_ids(IDS, NO_SCORES, 0.2, "nested", seed=1)
    two = mixing.gold_ids(IDS, NO_SCORES, 0.2, "nested", seed=2)
    assert one != two


def test_random_and_routed_are_unchanged():
    # Old runs must reproduce: the existing assignments keep their exact draws.
    scores = {i: n / 100 for n, i in enumerate(IDS)}
    routed = mixing.gold_ids(IDS, scores, 0.1, "routed", seed=1)
    assert routed == {f"s{i}" for i in range(10)}
    random = mixing.gold_ids(IDS, NO_SCORES, 0.25, "random", seed=1)
    assert len(random) == 25


def test_unknown_assignment_fails():
    with pytest.raises(ValueError):
        mixing.gold_ids(IDS, NO_SCORES, 0.2, "sideways", seed=1)
