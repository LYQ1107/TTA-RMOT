"""Offline evaluation grouping from official canonical source expressions only."""
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.multidomain_runtime import RUN,PUBLIC,SOURCE,require_plan
from scripts.multidomain_diagnostics import verify,write_csv
from flex_tta.runtime.pf_receipts import atomic_json,file_sha


def main():
    require_plan();verify('KITTI-V2','V2_COMPLETE_REFERENCE')
    meta=json.loads((RUN/'kitti/metadata.json').read_text());mapping=[]
    for q in meta['episodes']:
        p=SOURCE/'expression'/q['video_id']/(q['expression_id']+'.json')
        item=json.loads(p.read_text());text=item.get('raw_sentence',item['sentence']).lower()
        car=bool(re.search(r'\b(?:cars?|autos?|automobiles?|vehicles?)\b',text))
        pedestrian=bool(re.search(r'\b(?:pedestrains?|pedestrians?|people|persons?|males?|females?|men|women|womales)\b',text))
        if car==pedestrian:raise ValueError('Official canonical query category ambiguous: '+str(p))
        mapping.append(dict(video=q['video_id'],expression=q['expression_id'],canonical_text=text,
            canonical_field='raw_sentence' if 'raw_sentence' in item else 'sentence',category='car' if car else 'pedestrian',
            official_expression_path=str(p),official_expression_sha256=file_sha(p)))
    write_csv(PUBLIC/'source_query_categories.csv',mapping)
    atomic_json(RUN/'source_query_categories.json',dict(role='OFFLINE_EVALUATION_ONLY_AFTER_SEALED_V2_REFERENCE',
        mapping={r['video']+'+'+r['expression']:r['category'] for r in mapping},
        never_model_input=True,canonical_fields_only=True,referring_label_values_not_used=True,
        counts={c:sum(r['category']==c for r in mapping) for c in ('car','pedestrian')}))
    print(json.dumps(dict(event='OFFLINE_CATEGORY_MAP_COMPLETE',counts={c:sum(r['category']==c for r in mapping) for c in ('car','pedestrian')})),flush=True)


if __name__=='__main__':main()
