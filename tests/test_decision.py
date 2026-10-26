"""Decision tasks: one answer per input, from student and teacher alike."""

from active_gliner import tasks
from active_gliner.data.records import Record
from active_gliner.tasks import decision
from active_gliner.teachers import prompts, validate

LABELS = {"no": "no", "yes": "yes"}


def _record(labels):
    return Record(
        "boolq:en-US:train-0",
        None,
        "question: is it?\npassage: yes.",
        "decision",
        "en-US",
        "train",
        {"labels": labels},
    )


def test_training_example_is_single_label_and_skips_an_empty_answer():
    example = decision.to_training_example(_record(["yes"]), LABELS)
    task = example["output"]["classifications"][0]
    assert task["true_label"] == ["yes"] and task["multi_label"] is False
    assert decision.to_training_example(_record([]), LABELS) is None


def test_teacher_answer_must_be_one_allowed_option():
    record = _record(["yes"])
    gold, errors = validate.parse(record, "decision", '{"label": "no"}', LABELS)
    assert gold == {"labels": ["no"]} and errors == []
    gold, errors = validate.parse(record, "decision", '{"label": "maybe"}', LABELS)
    assert gold == {"labels": []} and errors[0]["kind"] == "label_not_allowed"
    gold, errors = validate.parse(record, "decision", "[]", LABELS)
    assert gold is None and errors[0]["kind"] == "schema"
    gold, errors = validate.parse(record, "decision", '{"label": ["yes", "no"]}', LABELS)
    assert gold is None and errors[0]["kind"] == "schema"


def test_prompt_schema_allows_exactly_one_option():
    schema = prompts.response_schema("decision", LABELS)
    assert schema["properties"]["label"]["enum"] == ["no", "yes"]
    assert schema["required"] == ["label"]


def test_wrong_answer_counts_as_one_false_positive_and_one_miss():
    gold = decision.gold_items(_record(["yes"]))
    pred = decision.pred_items({"labels": ["no"]})
    assert (len(gold & pred), len(pred - gold), len(gold - pred)) == (0, 1, 1)


def test_clinc_out_of_scope_gets_a_readable_name():
    pool = [
        Record("c:0", None, "hi", "decision", "en-US", "train", {"labels": ["oos"]}),
        Record("c:1", None, "x", "decision", "en-US", "train", {"labels": ["book_flight"]}),
    ]
    assert tasks.labels("clinc150", {"pool": pool}) == {
        "book_flight": "book flight",
        "oos": "out of scope",
    }
