"""Thin HTTP helpers shared by the OpenRouter and Ollama clients.

We talk to the REST APIs directly with httpx instead of vendor SDKs: two
providers, four endpoints, and no exposure to SDK churn.
"""

from __future__ import annotations

import random
import time
from typing import Any, Callable

import httpx

from .errors import BackendError

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
MAX_ATTEMPTS = 4
BASE_BACKOFF = 2.0


def _sleep_for(attempt: int, response: httpx.Response | None) -> float:
    if response is not None:
        retry_after = response.headers.get("retry-after")
        if retry_after:
            try:
                return min(60.0, float(retry_after))
            except ValueError:
                pass
    return min(32.0, BASE_BACKOFF * (2 ** (attempt - 1))) * (0.8 + random.random() * 0.4)


def request_with_retry(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    what: str,
    max_attempts: int = MAX_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
    **kwargs: Any,
) -> httpx.Response:
    """Issue a request, retrying transient failures with jittered backoff."""
    last_error: str = ""
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.request(method, url, **kwargs)
        except httpx.TimeoutException as exc:
            last_error = f"timed out after {client.timeout.read}s ({exc})"
            if attempt == max_attempts:
                break
            sleep(_sleep_for(attempt, None))
            continue
        except httpx.HTTPError as exc:
            last_error = str(exc)
            if attempt == max_attempts:
                break
            sleep(_sleep_for(attempt, None))
            continue

        if response.status_code < 400:
            return response

        last_error = f"HTTP {response.status_code}: {response.text[:400]}"
        if response.status_code in RETRY_STATUS and attempt < max_attempts:
            sleep(_sleep_for(attempt, response))
            continue
        raise BackendError(f"{what} failed — {last_error}", hint=_hint(response))

    raise BackendError(f"{what} failed after {max_attempts} attempts — {last_error}")


def _hint(response: httpx.Response) -> str | None:
    if response.status_code == 401:
        return "Check OPENROUTER_API_KEY — the key was rejected."
    if response.status_code == 402:
        return "Out of OpenRouter credits, or the request exceeds your spend limit."
    if response.status_code == 404:
        return "Model slug not found. Run `podsum models` to list what is live today."
    if response.status_code == 413:
        return "Payload too large. Lower --chunk-seconds so each upload stays small."
    return None


def json_body(response: httpx.Response, what: str) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError as exc:
        raise BackendError(
            f"{what} returned a non-JSON response: {response.text[:200]}"
        ) from exc
    if not isinstance(data, dict):
        raise BackendError(f"{what} returned an unexpected payload: {type(data).__name__}")
    if error := data.get("error"):
        message = error.get("message") if isinstance(error, dict) else str(error)
        raise BackendError(f"{what} returned an error: {message}")
    return data
