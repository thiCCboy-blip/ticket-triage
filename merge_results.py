import json
from datetime import datetime, timezone
from pathlib import Path

base = Path(__file__).resolve().parent
chunks = [json.loads((base / f"results_chunk{i}.json").read_text(encoding="utf-8")) for i in (1, 2, 3)]
results = sorted((r for c in chunks for r in c["results"]), key=lambda r: r["id"])
valid = sum(r["valid"] for r in results)
report = {
    "provider": "nvidia",
    "model": chunks[0]["model"],
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "total": len(results),
    "valid_json": valid,
    "results": results,
}
(base / "results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"{valid}/{len(results)} merged -> results.json")
