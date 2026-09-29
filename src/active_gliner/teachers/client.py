import hashlib
import ipaddress
import json
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from threading import Lock

import httpx

from active_gliner.data.records import Record

from . import prompts, validate
from .cache import LabelCache
from .config import TeacherConfig
from .stats import TeacherStats


class RateLimiter:
    def __init__(self, rpm: int | None):
        self.rpm = rpm
        self.times = deque()
        self.lock = Lock()

    def wait(self) -> None:
        if self.rpm is None:
            return
        with self.lock:
            while True:
                now = time.monotonic()
                while self.times and now - self.times[0] >= 60:
                    self.times.popleft()
                if len(self.times) < self.rpm:
                    self.times.append(now)
                    return
                time.sleep(max(0, 60 - (now - self.times[0])))


@dataclass
class LabelResult:
    labels: dict
    stats: dict


def _wait_for_server(client, cfg, delay) -> None:
    url = httpx.URL(cfg.base_url)
    local = url.host == "localhost"
    try:
        address = ipaddress.ip_address(url.host)
        local = address.is_loopback or address.is_private
    except ValueError:
        pass
    if not local:
        time.sleep(min(delay, cfg.max_wait_server_s))
        return
    health_url = cfg.base_url.rstrip("/").removesuffix("/v1") + "/health"
    deadline = time.monotonic() + cfg.max_wait_server_s
    while (remaining := deadline - time.monotonic()) > 0:
        try:
            response = client.get(health_url, timeout=min(5.0, remaining))
            if response.is_success:
                return
        except httpx.TransportError:
            pass
        time.sleep(min(max(cfg.retry_wait_s, 1.0), max(0, deadline - time.monotonic())))


def _request(client, body, cfg, limiter, stats, lock) -> dict:
    for attempt in range(cfg.max_retries + 1):
        limiter.wait()
        with lock:
            stats.requests += 1
        try:
            response = client.post(f"{cfg.base_url.rstrip('/')}/chat/completions", json=body)
            response.raise_for_status()
            return response.json()
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            kind = "http"
            if isinstance(exc, httpx.TransportError):
                kind = "timeout" if isinstance(exc, httpx.TimeoutException) else "transport"
            with lock:
                stats.errors_by_kind[kind] = stats.errors_by_kind.get(kind, 0) + 1
            retryable = isinstance(exc, httpx.TransportError) or (
                exc.response.status_code == 429 or 500 <= exc.response.status_code < 600
            )
            if not retryable or attempt == cfg.max_retries:
                raise
            with lock:
                stats.retries += 1
            delay = cfg.retry_wait_s * 2**attempt
            if isinstance(exc, httpx.TransportError):
                _wait_for_server(client, cfg, delay)
            else:
                time.sleep(delay)
    raise RuntimeError("Retry loop did not return")


def label_records(
    records: list[Record],
    task: str,
    labels: dict[str, str],
    cfg: TeacherConfig,
    cache_dir,
    transport=None,
    dataset: str = "unknown",
    definitions: dict | None = None,
    prompt_version: str | None = None,
) -> LabelResult:
    phase_started = time.monotonic()
    headers = {}
    if cfg.api_key_env:
        api_key = os.environ.get(cfg.api_key_env)
        if not api_key:
            raise ValueError(
                f"Set {cfg.api_key_env} in the environment or in .env (never commit it)"
            )
        headers["Authorization"] = f"Bearer {api_key}"
    version = prompt_version or (
        "v2" if dataset == "unknown" and definitions is not None else prompts.version_for(dataset)
    )
    if version not in {"v1", "v2"}:
        raise ValueError(f"Unknown prompt version: {version}")
    if version == "v1":
        definitions = None
    elif definitions is None:
        definitions = prompts.definitions_for(dataset)
    schema = prompts.response_schema(task, labels)
    digest = prompts.prompt_hash(task, labels, definitions)
    if any(record.task != task for record in records):
        raise ValueError("Every record must match the labelling task")
    caches = {
        locale: LabelCache.for_job(cache_dir, cfg.name, task, dataset, locale, digest)
        for locale in {record.locale for record in records}
    }
    stats, lock = TeacherStats(workers=cfg.workers, gpu_shared=cfg.gpu_shared), Lock()
    limiter = RateLimiter(cfg.rpm_limit)
    results = {}

    with httpx.Client(transport=transport, timeout=cfg.timeout_s, headers=headers) as client:

        def label(record):
            cache = caches[record.locale]
            messages = prompts.messages(
                task, record.text, labels, definitions=definitions, record=record
            )
            input_hash = hashlib.sha256(
                json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()
            cached = cache.get(record.id)
            if cached is not None and cached.get("input_hash") == input_hash:
                counts = {}
                gold, errors = validate.parse(record, task, cached["raw"], labels, counts)
                return (
                    record.id,
                    {
                        **cached,
                        "gold": gold,
                        "errors": errors,
                        "duplicate": counts.get("duplicate", 0),
                    },
                    True,
                )
            body = {
                "model": cfg.model,
                "messages": messages,
                "temperature": cfg.temperature,
                "max_tokens": cfg.max_tokens,
            }
            if cfg.send_template_kwargs:
                body["chat_template_kwargs"] = {"enable_thinking": cfg.thinking}
            if cfg.json_schema:
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": task, "schema": schema},
                }
            body.update(cfg.extra_body)
            started = time.monotonic()
            # Failed requests must stay out of the cache so a rerun retries them.
            response = _request(client, body, cfg, limiter, stats, lock)
            latency = time.monotonic() - started
            usage = response.get("usage") or {}
            try:
                raw = response["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise ValueError("Teacher response has no message content") from exc
            counts = {}
            gold, errors = validate.parse(record, task, raw, labels, counts)
            value = {
                "duplicate": counts.get("duplicate", 0),
                "input_hash": input_hash,
                "gold": gold,
                "errors": errors,
                "raw": raw,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "latency_s": latency,
            }
            cache.put(record.id, value)
            return record.id, value, False

        with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
            futures = [pool.submit(label, record) for record in records]
            for future in as_completed(futures):
                try:
                    record_id, value, cached = future.result()
                except (httpx.TransportError, httpx.HTTPStatusError):
                    with lock:
                        kind = "request_failed"
                        stats.errors_by_kind[kind] = stats.errors_by_kind.get(kind, 0) + 1
                        stats.invalid_records += 1
                    continue
                results[record_id] = value
                stats.cache_hits += int(cached)
                stats.duplicate += value.get("duplicate", 0)
                if not cached:
                    stats.prompt_tokens += value["prompt_tokens"]
                    stats.completion_tokens += value["completion_tokens"]
                    stats.latency_s_total += value["latency_s"]
                stats.invalid_records += int(value["gold"] is None or bool(value["errors"]))
                for kind in [error["kind"] for error in value["errors"]]:
                    with lock:
                        stats.errors_by_kind[kind] = stats.errors_by_kind.get(kind, 0) + 1
    stats.elapsed_s = time.monotonic() - phase_started
    stats.valid_rate = 1.0 - stats.invalid_records / (len(records) or 1)
    stats.cost_usd = (
        stats.prompt_tokens * cfg.price_in_per_m + stats.completion_tokens * cfg.price_out_per_m
    ) / 1e6
    return LabelResult(results, stats.to_dict())
