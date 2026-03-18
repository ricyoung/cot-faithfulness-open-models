import json
from pathlib import Path

results_dir = Path("results")

for base_file in sorted(results_dir.glob("base/*.jsonl")):
    if "errors" in base_file.name:
        continue
    model = base_file.stem
    try:
        lines = open(base_file, encoding="utf-8", errors="replace").readlines()
        results = [json.loads(l) for l in lines if l.strip()]
    except Exception as e:
        print(f"CORRUPT FILE: {base_file} - {e}")
        continue

    if not results:
        continue

    has_thinking = sum(1 for r in results if r.get("thinking_text", ""))
    has_answer = sum(1 for r in results if r.get("extracted_answer"))
    has_tokens = sum(1 for r in results if r.get("reasoning_tokens", 0) > 0)
    correct = sum(1 for r in results if r.get("extracted_answer") == r.get("correct_label"))
    avg_latency = sum(r.get("latency_seconds", 0) for r in results) / len(results)

    print(f"{model}:")
    print(f"  Questions: {len(results)}")
    print(f"  Thinking text: {has_thinking}/{len(results)} ({100*has_thinking//len(results)}%)")
    print(f"  Extracted answer: {has_answer}/{len(results)} ({100*has_answer//len(results)}%)")
    print(f"  Reasoning tokens: {has_tokens}/{len(results)} ({100*has_tokens//len(results)}%)")
    print(f"  Accuracy: {correct}/{len(results)} ({100*correct/len(results):.1f}%)")
    print(f"  Avg latency: {avg_latency:.1f}s")
    print()

# Check errors
for err_file in sorted(results_dir.glob("base/*.errors.jsonl")):
    try:
        errors = [json.loads(l) for l in open(err_file, encoding="utf-8", errors="replace") if l.strip()]
    except Exception:
        continue
    if errors:
        model = err_file.stem.replace(".errors", "")
        print(f"ERRORS - {model}: {len(errors)} errors")
        print(f"  Sample: {errors[0]['error'][:150]}")
        print()
