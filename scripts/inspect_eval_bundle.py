# -*- coding: utf-8 -*-
"""Extract full details of the 41 extracted paper cards from latest run."""
import json, sys, os
sys.stdout.reconfigure(encoding='utf-8')

eval_dir = max(
    [os.path.join("data/eval_bundles", d) for d in os.listdir("data/eval_bundles")],
    key=os.path.getmtime
)
print(f"Eval bundle: {eval_dir}")

# Look inside eval_bundle files
for fname in os.listdir(eval_dir):
    fpath = os.path.join(eval_dir, fname)
    size = os.path.getsize(fpath)
    print(f"  {fname} ({size} bytes)")

# Read metadata.json
with open(os.path.join(eval_dir, "metadata.json"), "r", encoding="utf-8") as f:
    meta = json.load(f)
print(f"\nMetadata: {json.dumps(meta, ensure_ascii=False, indent=2)}")

# Read global_evidence_gate.json
with open(os.path.join(eval_dir, "global_evidence_gate.json"), "r", encoding="utf-8") as f:
    gate = json.load(f)
print(f"\nGate Details: {json.dumps(gate, ensure_ascii=False, indent=2)}")

# Read final_routes.json
with open(os.path.join(eval_dir, "final_routes.json"), "r", encoding="utf-8") as f:
    routes = json.load(f)
print(f"\nRoutes Details: {json.dumps(routes, ensure_ascii=False, indent=2)}")
