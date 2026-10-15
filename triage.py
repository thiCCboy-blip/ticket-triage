#!/usr/bin/env python3
"""Support-ticket triage bot (multi-provider).

Sends each support ticket to an LLM with a fixed system prompt and parses back
JSON: {"category", "urgency", "summary"}. Works with any OpenAI-compatible
API (OpenAI, NVIDIA NIM, Groq, OpenRouter...) plus Anthropic's Claude API.

Usage:
  python triage.py --mock                    # offline pipeline check, no key needed
  python triage.py --provider nvidia         # NVIDIA NIM (needs NVIDIA_API_KEY)
  python triage.py --provider openai         # OpenAI (needs OPENAI_API_KEY)
  python triage.py --provider anthropic      # Claude (needs ANTHROPIC_API_KEY)
  python triage.py --provider nvidia --ids 3,17,46
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
TICKETS_PATH = BASE_DIR / "tickets.json"
SYSTEM_PROMPT_PATH = BASE_DIR / "system_prompt.txt"
DEFAULT_OUT = BASE_DIR / "results.json"

PROVIDERS = {
    "anthropic": {
        "key_env": "ANTHROPIC_API_KEY",
        "base_url": "https://api.anthropic.com",
        "model": "claude-sonnet-4-5",
    },
    "openai": {
        "key_env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    },
    "nvidia": {
        "key_env": "NVIDIA_API_KEY",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "model": "openai/gpt-oss-20b",
    },
}

CATEGORIES = {
    "technical", "billing", "account", "shipping", "product_defect",
    "feature_request", "complaint", "off_topic", "other",
}
URGENCIES = {"critical", "high", "medium", "low"}
VALID_KEYS = {"category", "urgency", "summary"}
MAX_SUMMARY_WORDS = 12


def load_tickets() -> list[dict]:
    return json.loads(TICKETS_PATH.read_text(encoding="utf-8"))


def extract_json(text: str) -> dict | None:
    if not text or not text.strip():
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start == -1:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def validate(obj) -> list[str]:
    if not isinstance(obj, dict):
        return ["response is not a JSON object"]
    errors = []
    if set(obj.keys()) != VALID_KEYS:
        errors.append(f"keys must be exactly {sorted(VALID_KEYS)}, got {sorted(obj.keys())}")
    if obj.get("category") not in CATEGORIES:
        errors.append(f"invalid category: {obj.get('category')!r}")
    if obj.get("urgency") not in URGENCIES:
        errors.append(f"invalid urgency: {obj.get('urgency')!r}")
    summary = obj.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        errors.append("summary must be a non-empty string")
    elif "\n" in summary:
        errors.append("summary must be a single line")
    elif len(summary.split()) > MAX_SUMMARY_WORDS:
        errors.append(f"summary exceeds {MAX_SUMMARY_WORDS} words")
    return errors


def ticket_user_message(ticket: dict) -> str:
    return f"<ticket>\nSubject: {ticket['subject']}\n\n{ticket['body']}\n</ticket>"


def triage_anthropic(ticket: dict, system_prompt: str, model: str, api_key: str, retries: int):
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    for attempt in range(retries + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=300,
                temperature=0,
                system=system_prompt,
                messages=[{"role": "user", "content": ticket_user_message(ticket)}],
            )
            raw = "".join(block.text for block in response.content if block.type == "text")
            obj = extract_json(raw)
            if obj is not None:
                return obj, raw
            time.sleep(1)
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
            if attempt == retries:
                raise RuntimeError(f"API failed after {retries} retries: {exc}") from exc
            time.sleep(2 ** attempt)


def openai_compatible_call(base_url: str, api_key: str, payload: dict) -> str:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body["choices"][0]["message"].get("content") or ""


def triage_openai_compatible(ticket: dict, system_prompt: str, model: str, base_url: str,
                             api_key: str, retries: int):
    payload = {
        "model": model,
        "max_tokens": 2048,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ticket_user_message(ticket)},
        ],
    }
    last_error = None
    for attempt in range(retries + 1):
        try:
            raw = openai_compatible_call(base_url, api_key, payload)
            obj = extract_json(raw)
            if obj is not None:
                return obj, raw
            last_error = f"no JSON in model output: {raw[:120]!r}"
            time.sleep(1)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            if exc.code == 400 and "temperature" in detail and "temperature" in payload:
                del payload["temperature"]
                continue
            last_error = f"HTTP {exc.code}: {detail}"
            if exc.code in (401, 403, 404):
                break
            time.sleep(2 ** attempt)
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"API failed after {retries} retries: {last_error}")


def triage_mock(ticket: dict) -> dict:
    text = f"{ticket['subject']}\n{ticket['body']}".lower()
    category = "other"
    for name, keywords in MOCK_CATEGORY_KEYWORDS:
        if any(k in text for k in keywords):
            category = name
            break
    if any(k in text for k in MOCK_CRITICAL):
        urgency = "critical"
    elif any(k in text for k in MOCK_HIGH):
        urgency = "high"
    elif category in ("off_topic", "feature_request", "complaint") or any(k in text for k in MOCK_LOW):
        urgency = "low"
    else:
        urgency = "medium"
    if category == "off_topic":
        summary = "Not a support request; auto-classified from message content"
    elif urgency == "medium" and len(ticket["body"]) < 40:
        summary = "Vague report; needs product area and error details"
    else:
        words = ticket["subject"].replace(".", "").split()[:12]
        summary = " ".join(words)
    return {"category": category, "urgency": urgency, "summary": summary}


MOCK_OFF_TOPIC = (
    "resume", "hiring", "senior developer role", "seo", "backlink", "pharmacy",
    "prescription", "charity", "gala", "wedding", "bouquet", "restaurant",
    "currywurst", "netflix", "election", "sponsor a table", "dog's name",
    "increase your revenue",
)
MOCK_CATEGORY_KEYWORDS = [
    ("off_topic", MOCK_OFF_TOPIC),
    ("billing", ("charge", "charged", "invoice", "refund", "billing", "payment", "prorated", "annual", "cancel")),
    ("shipping", ("delivery", "delivered", "tracking", "package", "dispatch", "shipped", "headset")),
    ("account", ("password", "login", "locked", "2fa", "admin", "account", "can't get in", "sessions from ip")),
    ("product_defect", ("leaking", "leak", "swollen", "sparked", "shattered", "burn")),
    ("technical", ("error", "500", "crash", "crashed", "broken", "performance", "slow", "seconds", "outage", "down", "service unavailable", "update broke", "not working")),
    ("feature_request", ("dark mode", "please add", "feature")),
    ("complaint", ("disgusted", "furious", "worst", "sucks", "ruined", "idiot", "lying")),
]
MOCK_CRITICAL = ("security", "another country", "not mine", "breach", "sparked", "fire hazard", "burned my", "all 40 team", "every user in our tenant", "swollen")
MOCK_HIGH = ("chargeback", "lawyer", "legal", "gdpr", "refund it", "money back", "locked me out", "cannot work", "40k followers", "consumer protection", "overcharged", "eaten my work", "lost two hours", "blocked", "urgent", "asap", "written confirmation", "different customer")
MOCK_LOW = ("icon", "color", "dark mode", "recommendation", "restaurant", "charity", "gala")


def resolve_provider(provider_flag: str) -> str:
    if provider_flag != "auto":
        return provider_flag
    for name, cfg in PROVIDERS.items():
        if os.environ.get(cfg["key_env"]):
            return name
    sys.exit("No API key found. Set NVIDIA_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY (or run with --mock).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Triage support tickets into category/urgency/summary JSON.")
    parser.add_argument("--mock", action="store_true", help="run offline heuristic triage; no API calls")
    parser.add_argument("--provider", choices=["auto", *PROVIDERS], default="auto",
                        help="API provider (default: auto-detect from env keys)")
    parser.add_argument("--base-url", help="override base URL for any OpenAI-compatible endpoint")
    parser.add_argument("--ids", help="comma-separated ticket ids to process, e.g. 3,17,46")
    parser.add_argument("--limit", type=int, help="process only the first N tickets")
    parser.add_argument("--model", help="model id (default: provider-specific)")
    parser.add_argument("--retries", type=int, default=2, help="retries per ticket on API or parse failure")
    parser.add_argument("--delay", type=float, default=0.2, help="seconds to sleep before each API call")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="results file path")
    parser.add_argument("--verbose", action="store_true", help="print raw model output on failures")
    args = parser.parse_args()

    provider = "mock" if args.mock else resolve_provider(args.provider)
    if provider != "mock":
        cfg = PROVIDERS[provider]
        api_key = os.environ.get(cfg["key_env"])
        if not api_key:
            sys.exit(f"{cfg['key_env']} is not set. Get a free key and export it, or run with --mock.")
        model = args.model or cfg["model"]
        base_url = (args.base_url or cfg["base_url"]).rstrip("/")
        if provider == "anthropic":
            try:
                import anthropic  # noqa: F401
            except ModuleNotFoundError:
                sys.exit("Missing dependency for Anthropic. Run: pip install -r requirements.txt  (or use another --provider)")
    else:
        api_key, model, base_url = None, "mock", None

    tickets = load_tickets()
    if args.ids:
        wanted = {int(x) for x in args.ids.split(",")}
        tickets = [t for t in tickets if t["id"] in wanted]
        if not tickets:
            sys.exit(f"No tickets match ids {sorted(wanted)}")
    if args.limit:
        tickets = tickets[: args.limit]

    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    results = []
    valid_count = 0

    for ticket in tickets:
        raw = None
        error = None
        if provider == "mock":
            obj = triage_mock(ticket)
        else:
            if args.delay:
                time.sleep(args.delay)
            try:
                if provider == "anthropic":
                    obj, raw = triage_anthropic(ticket, system_prompt, model, api_key, args.retries)
                else:
                    obj, raw = triage_openai_compatible(ticket, system_prompt, model, base_url, api_key, args.retries)
            except Exception as exc:
                obj, error = None, str(exc)
        errors = validate(obj) if error is None else [error]
        ok = not errors
        valid_count += ok
        results.append({
            "id": ticket["id"],
            "tag": ticket["tag"],
            "valid": ok,
            "result": obj,
            "errors": errors,
        })
        status = "VALID  " if ok else "INVALID"
        shown = obj["summary"][:70] if isinstance(obj, dict) and isinstance(obj.get("summary"), str) else ""
        line = f"[{ticket['id']:>2}] {status}  {ticket['tag']:<9} "
        if ok:
            line += f"{obj['category']:<15} {obj['urgency']:<9} {shown}"
        else:
            line += "; ".join(errors)[:120]
        print(line)
        if args.verbose and raw and not ok:
            print(f"     raw: {raw!r}")

    report = {
        "provider": provider,
        "model": model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total": len(tickets),
        "valid_json": valid_count,
        "results": results,
    }
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{valid_count}/{len(tickets)} tickets returned valid JSON  ->  {args.out}")
    sys.exit(0 if valid_count == len(tickets) else 1)


if __name__ == "__main__":
    main()
