"""Preserve offsets through GLiNER 2.0.0's string-only training interface."""

from copy import deepcopy

from gliner2.processor import SchemaTransformer

from active_gliner.tasks._spans import ExactValue, exact_token_span


class ExactTargetProcessor(SchemaTransformer):
    def _collate_batch(self, batch, max_len=None, error_policy="raise"):
        # The parent appends punctuation, which can change a final URL's token boundary.
        records = [
            self._transform_record({"text": text, "schema": deepcopy(schema)}, max_len=max_len)
            for text, schema in batch
        ]
        return self._pad_batch(records)

    def _build_outputs(self, processed, schema, text_tokens, len_prefix):
        outputs = super()._build_outputs(processed, schema, text_tokens, len_prefix)
        for output, struct_label in zip(outputs, processed["structure_labels"], strict=True):
            if output["task_type"] == "classifications":
                continue
            _, spans = struct_label
            for fields, positions in zip(spans, output["output"][1], strict=True):
                for i, field in enumerate(fields):
                    values = field if isinstance(field, list) else [field]
                    if not values or not all(isinstance(v, ExactValue) for v in values):
                        continue
                    exact = [exact_token_span(v, self.word_splitter) for v in values]
                    if any(end + len_prefix >= len(text_tokens) for _, end in exact):
                        raise ValueError("Truncation would remove an exact training span")
                    positions[i] = [(start + len_prefix, end + len_prefix) for start, end in exact]
        return outputs

    def _process_json_structures(self, schema, schemas, labels, types, sampling):
        # Upstream synthetic slot fields look up renamed keys in the original mapping.
        # Disable that augmentation because it turns every supplied value into None.
        from dataclasses import replace

        if sampling:
            sampling = replace(sampling, synthetic_entity_label_prob=0.0)
        return super()._process_json_structures(schema, schemas, labels, types, sampling)


def install_exact_targets(student) -> None:
    processor = ExactTargetProcessor.__new__(ExactTargetProcessor)
    processor.__dict__.update(student.processor.__dict__)
    student.processor = processor
