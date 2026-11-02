"""Post-seal cross-domain Stage-A comparisons; no model or teacher calls."""
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_uda.presence.runtime import PUBLIC
from scripts.active_run import PAIRS
from scripts.presence_finish_a import references
from scripts.active_eval import bootstrap
from scripts.pf_diag_common import official_metrics
from scripts.diag_attrib_v2_common import write_csv
from scripts.presence_redesign_report import table
from flex_tta.runtime.pf_receipts import atomic_json,file_sha


def main():
    record=PUBLIC / 'redesign_v5'
    with (record / 'stage_A.csv').open() as h:all_rows=list(csv.DictReader(h))
    rows=[];intervals={};refs=[]
    for pair,(model,target) in PAIRS.items():
        r=next(r for r in all_rows if r['target']==target and r['model']==model and r['method']=='PRES_TOPK')
        reference=dict(references(target,model))
        interval=bootstrap(r['path'],reference['NATIVE'],target);intervals[pair]=interval
        oracle=official_metrics(reference['R_SxK_ORACLE'] / 'results')
        rows.append(dict(pair=pair,target=target,model=model,native_HOTA=float(r['native_HOTA']),
            v5_HOTA=float(r['HOTA']),delta_HOTA=interval['delta_HOTA'],CI95=interval['CI95'],
            quantity_ORACLE_HOTA=oracle['HOTA'],prediction_rows=int(r['prediction_rows']),
            added_rows=int(r['added_prediction_rows_vs_native']),removed_rows=int(r['removed_prediction_rows_vs_native']),
            query_empty_prediction_rows=int(r['query_empty_prediction_rows']),GT_open_attempts=int(r['GT_open_attempts'])))
        for name,folder in reference.items():
            seal=folder / 'predictions_seal.json'
            refs.append(dict(pair=pair,reference=name,path=str(folder),prediction_seal_sha256=file_sha(seal),
                oracle=name!='NATIVE'))
    write_csv(record / 'cross_domain.csv',rows)
    atomic_json(record / 'cross_bootstrap.json',dict(comparisons=intervals,samples=2000,seed=0,
        cluster_unit='video',KITTI='four videos, points only',purpose='descriptive; no target-GT parameter/variant selection'))
    atomic_json(record / 'reference_paths.json',dict(files=refs))
    path=record / 'REPORT.md';text=path.read_text();marker='\n## 7. 六跨域固定结果'
    if marker in text:text=text[:text.index(marker)]
    text+=marker+'\n\n'+table(list(rows[0]),rows)
    text+=('\n六方向五个点值上升、C6下降，不挑方向/平滑或更改两个源参数。'
        '本轮没有新增视觉语言语义或改变同帧排序；raw输出变化是数量增加，不是恢复独立的新排序信号。'
        'ORACLE列读取GT数量，仅作历史诊断上限；不得作为部署结果或训练输入。'
        '25/45视频聚类区间为事后描述，不把相关表达/帧当独立样本；KITTI只报点值。'
        '所有组合先封存后官方评测，生成GT尝试0、新GPU/教师/LoRA0。'
        '阶段A通过后继续已授权C1/C3真实pilot与40小时成本关卡，不把此表当完整方法已完成。\n')
    path.write_text(text)
    receipt=record / 'report_receipt.json';r=json.loads(receipt.read_text())
    r.update(report_sha256=file_sha(path),cross_summary_sha256=file_sha(record / 'cross_domain.csv'),
        cross_bootstrap_sha256=file_sha(record / 'cross_bootstrap.json'))
    atomic_json(receipt,r)
    print(json.dumps(dict(cross_combinations=len(rows),positive_point_deltas=sum(r['delta_HOTA']>0 for r in rows),
        new_teacher_GPU_calls=0)),flush=True)


if __name__=='__main__':main()
