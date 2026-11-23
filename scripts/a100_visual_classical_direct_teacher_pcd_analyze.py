"""Native-PCD direct teacher audit, including fixed RGB-direct references."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.a100_visual_classical_direct_teacher_analyze import main
from scripts.a100_visual_classical_direct_teacher_pcd_launch import conditions
from scripts.a100_visual_classical_direct_teacher_pcd import METHODS


def rgb_references(record):
    slug=record['pair']+f"_seed{record['seed']}"
    return {'RGB_'+name:dict(file='direct_teacher_parent_statistics.json',record=slug+'/'+name) for name in METHODS}


if __name__=='__main__':
    extra={'JOINT_vs_RGB_'+name:('JOINT','RGB_'+name) for name in METHODS}
    extra.update({'RES_JOINT_vs_RGB_'+name:('RES_JOINT','RGB_'+name) for name in ('KD400','KD_FINAL','KD_MEAN','HARD400','HARD_FINAL')})
    extra.update(KD_MEAN_vs_RGB_KD_MEAN=('KD_MEAN','RGB_KD_MEAN'),
        RGB_JOINT_vs_KD_MEAN=('RGB_JOINT','KD_MEAN'),
        RGB_JOINT_vs_JOINT=('RGB_JOINT','JOINT'))
    main(stem='direct_teacher_pcd',records=conditions(),dimension=768,
        extra_references=rgb_references,extra_contrasts=extra)
