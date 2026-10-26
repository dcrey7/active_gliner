"""UNER EWT: the tag is the third column, comments carry the document id."""

from active_gliner import tasks
from active_gliner.data import uner_ewt

SAMPLE = (
    "# newdoc id = answers-1\n"
    "# sent_id = answers-1-0001\n"
    "# text = Where is New York?\n"
    "1\tWhere\tO\t-\t-\n"
    "2\tis\tO\t-\t-\n"
    "3\tNew\tB-LOC\t-\tann\n"
    "4\tYork\tI-LOC\t-\tann\n"
    "5\t?\tO\t-\t-\n"
    "\n"
    "# sent_id = answers-1-0002\n"
    "# text = Thanks .\n"
    "1\tThanks\tO\t-\t-\n"
    "2\t.\tO\t-\t-\n"
    "\n"
)


def test_rows_read_the_third_column_and_keep_empty_sentences(tmp_path):
    for split in ("train", "dev", "test"):
        (tmp_path / f"en_ewt-ud-{split}.iob2").write_text(SAMPLE, encoding="utf-8")
    splits = uner_ewt.load(tmp_path, "en-US")
    first, second = splits["train"]
    assert first.doc_id == "answers-1" and first.task == "ner"
    assert first.gold["spans"] == [{"start": 9, "end": 17, "label": "LOC", "text": "New York"}]
    # A sentence with no entity stays in the data (research rule 4).
    assert second.gold["spans"] == [] and second.doc_id == "answers-1"


def test_uner_uses_the_ner_task():
    assert tasks.for_dataset("uner_ewt").__name__.endswith(".ner")
