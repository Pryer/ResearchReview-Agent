"""验证独立人工签署与原文，不按前端complete或human_gold直接验收。"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.services.field_review_validation import validate_field_review

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--template',required=True,type=Path)
    p.add_argument('--review',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    result=validate_field_review(json.loads(args.template.read_text(encoding='utf-8-sig')),
        json.loads(args.review.read_text(encoding='utf-8-sig')),sha256(args.template.read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='tasks'},ensure_ascii=True))
    raise SystemExit(0 if result['valid'] else 2)
