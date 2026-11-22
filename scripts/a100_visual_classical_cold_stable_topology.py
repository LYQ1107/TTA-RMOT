"""Fresh one-SE deployment using the frozen physical stream implementation."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import a100_visual_classical_cold_predictive_topology as engine
from flex_uda.active.topology_stability import teacher_stability_choice
from flex_uda.active.topology_selection import apply_graph


def selected_graph_inference(unary,pcd,colours,texts,answers,tracks,video,protocol):
    geometry,local,global_,supported,selected,_,receipt=teacher_stability_choice(pcd,colours,texts,answers,tracks,video,protocol)
    return apply_graph(unary,answers,geometry,(local,global_)[selected],supported),receipt


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--name',required=True);parser.add_argument('--pair',choices=('C4','C1'),required=True)
    parser.add_argument('--perception-mode',default='track-three');parser.add_argument('--methods',default='CLIP_MEAN_COHORT_CAL')
    parser.add_argument('--teacher-fraction',type=float,default=.1);parser.add_argument('--query-graph-scope',default='extended')
    parser.add_argument('--head-query-scope',default='pcd');parser.add_argument('--task-protocol',required=True)
    parser.add_argument('--set-decoder',default='original');parser.add_argument('--graph-metric',default='combined')
    parser.add_argument('--graph-prior',default='constant');parser.add_argument('--head-source-logit',default='zero')
    args=parser.parse_args()
    # Explicit inference dependency: acquisition, neural fitting, teacher and
    # all timing/output code stay in the already verified frozen stream.
    engine.selected_graph_inference=selected_graph_inference
    engine.run(args.name,args.perception_mode,args.methods.split(','),args.teacher_fraction,args.pair,args.query_graph_scope,
        args.head_query_scope,args.task_protocol,args.set_decoder,args.graph_metric,args.graph_prior,args.head_source_logit)


if __name__=='__main__':main()
