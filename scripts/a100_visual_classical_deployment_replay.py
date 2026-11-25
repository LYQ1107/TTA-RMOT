"""GT-free regression against sealed experimental heads and set decoders."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.special import logit
from flex_tta.runtime.pf_receipts import atomic_json
from flex_uda.a100.mechanism import generation_guard,metadata,PAIRS
from flex_uda.active.context_cohort import distillation_features
from flex_uda.active.task_decode import decode_with_contract
from flex_uda.active.acquisition import exclusive_group
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_rgb_parity import PARITY,source_spec
from scripts.a100_visual_classical_zero_colour import ZERO
from scripts.a100_visual_classical_null_set import NULL
from scripts.a100_visual_classical_strict_prefix import plain


def main():
    import torch
    from torch import nn
    guard=generation_guard();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    checks=[];decoder_checks=0;decoder_changed=0
    for pair in ('C4','C1'):
        target=PAIRS[pair][1];m=metadata(target);source=source_spec(pair)[0];ref=PARITY/pair/'seed0';video=sorted(m['videos'])[-1]
        with np.load(source/'source'/video/'sparse_inputs.npz') as a:pcd,colours,prior,tracks=a['pcd'],a['colours'],a['prior'],a['tracks']
        bank=m['videos'][video];queries=[q for q in m['episodes'] if q['video_id']==video];texts=[q['query_text'] for q in queries]
        obs=np.asarray(bank['observations']);lookup={int(t):i for i,t in enumerate(tracks)};cols=np.asarray([lookup[int(t)] for _,t in obs])
        for scope in ('clothing','extended','zero'):
            if scope=='zero':checkpoint=ZERO/pair/'seed0/ZERO_latest_head.pt';expected=ZERO/pair/'seed0/ZERO_FINAL/videos'/video/'mask.npy'
            else:
                label='ORIG' if scope=='clothing' and pair=='C4' else 'RGB'
                checkpoint=ref/(label+'_latest_head.pt');expected=ref/(label+'_KD_FINAL')/'videos'/video/'mask.npy'
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False);x=distillation_features(pcd,colours,texts,scope).reshape(-1,1080)
            model=nn.Sequential(nn.Linear(1080,64),nn.GELU(),nn.Linear(64,1));model.load_state_dict(saved['model']);model.to('cuda:0').eval()
            values=torch.tensor((x-saved['mean'])/saved['std'],dtype=torch.float32,device='cuda:0')
            z=torch.tensor(np.clip(logit(np.clip(prior,.05,.95)),-2,2).ravel(),dtype=torch.float32,device='cuda:0')
            with torch.no_grad():score=torch.sigmoid(z+model(values).squeeze(-1)).cpu().numpy().reshape(prior.shape)
            mask=plain(score[:,cols],exclusive_group(texts,target));changed=int(np.count_nonzero(mask!=np.load(expected)));assert changed==0,(pair,scope,changed)
            checks.append(dict(pair=pair,scope=scope,video=video,changed_candidate_rows=changed));del model,values,z
        for video,bank in sorted(m['videos'].items()):
            obs=np.asarray(bank['observations']);base=np.load(ref/'RGB_KD_MEAN/videos'/video/'mask.npy');graph=np.load(ref/'RGB_GRAPH_MEAN/videos'/video/'mask.npy')
            reference=np.load(ref/'RGB_KD_MEAN/videos'/video/'scores.npy');gscore=np.load(ref/'RGB_GRAPH_MEAN/videos'/video/'scores.npy')
            with np.load(source/'source'/video/'sparse_inputs.npz') as a:tracks,prior=a['tracks'],a['prior']
            for mode,refined,name in [('shared_singleton',False,'KD_MEAN_PRESENCE'),('shared_singleton',True,'GRAPH_MEAN_PRESENCE'),('retain_distilled_set',True,'SAFE_MEAN')]:
                before=graph if refined else base;score=gscore if refined else reference
                result,_,_=decode_with_contract(score,before,base,reference,obs,tracks,prior,
                    'PRESUPPOSED' if pair=='C4' else 'CHECKLIST',mode,refined)
                expected=np.load(NULL/'seed0/C4'/name/'videos'/video/'mask.npy') if pair=='C4' else before
                changed=int(np.count_nonzero(result!=expected));assert changed==0,(pair,video,mode,changed)
                decoder_changed+=changed;decoder_checks+=1
    assert not guard
    result=dict(status='PASS_DEPLOYMENT_INPUT_AND_SET_DECODER_REPLAY',head_last_video_checks=checks,
        full_video_decoder_checks=decoder_checks,decoder_changed_candidate_rows=decoder_changed,GT_open_attempts=0,
        source_or_teacher_forward_calls=0,new_head_updates=0,saved_teacher_trained_heads_only=True,
        cold_runtime_defaults_preserve_original_path=True,not_a_new_quality_or_cold_cost_measurement=True)
    atomic_json(ROOT/'docs/a100_visual_classical_20261121/deployment_input_and_decoder_replay.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
