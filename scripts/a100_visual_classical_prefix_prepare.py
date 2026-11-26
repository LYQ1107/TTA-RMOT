"""Split recorded frozen-teacher answers into strictly visible policy/video files."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.mechanism import generation_guard
from scripts.a100_visual_classical_prepare import OUT


def run():
    attempts=generation_guard();summary={}
    for pair in ('C4','C1'):
        source=OUT/'evidence'/pair
        responses={(r['video_id'],r['local_query'],r['local_track']):r for r in map(json.loads,(source/'responses.jsonl').read_text().splitlines())}
        plan=json.loads((source/'acquisition_plan.json').read_text())['positions']
        evidence_sha=file_sha(source/'responses.jsonl')
        counts={}
        for video,policies in plan.items():
            for policy,positions in policies.items():
                rows=[responses[video,int(q),int(t)] for q,t in positions]
                path=OUT/'causal_teacher'/pair/policy/(video+'.json')
                atomic_json(path,dict(pair=pair,policy=policy,video=video,answers=rows,logical_calls=len(rows),
                    target_GT_used=False,only_this_video_and_policy=True,source_evidence_sha256=evidence_sha))
                counts[policy]=counts.get(policy,0)+len(rows)
        summary[pair]=counts
    assert not attempts
    atomic_json(OUT/'causal_teacher/completion.json',dict(status='VISIBLE_TEACHER_SHARDS_READY',
        counts=summary,GT_open_attempts=0,prepaid_recording_not_a_live_wall_benchmark=True))


if __name__=='__main__':run()
