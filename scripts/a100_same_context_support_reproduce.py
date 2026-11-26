"""Independent set-intersection/NetworkX check of metadata comparison support."""
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import networkx as nx
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.mechanism import PAIRS,metadata,generation_guard,KEYS
from flex_uda.active.native_binary_inputs import inventory,PREP
from scripts.a100_six_direction import OUT as SIX,PUBLIC


def graph_from_track_contexts(tracks,contexts):
    graph=nx.Graph();graph.add_nodes_from(tracks)
    for a,b in combinations(sorted(contexts),2):
        if contexts[a]&contexts[b]:graph.add_edge(a,b)
    parts=list(nx.connected_components(graph))
    return graph,dict(quotient_components=len(parts),quotient_rank=len(tracks)-len(parts),
        unique_quotient_edges=graph.number_of_edges(),pairable_tracks=sum(graph.degree(t)>0 for t in tracks),
        largest_component_tracks=max(map(len,parts)))


def main():
    guard=generation_guard();path=PUBLIC/'same_context_comparison_support_20261124.json'
    primary=json.loads(path.read_text());data=json.loads(Path(primary['local_details']).read_text())
    assert file_sha(Path(primary['local_details']))==primary['local_details_sha256']
    physical={(r['target'],r['video']):r for r in data['physical_videos']}
    query_rows={(r['pair'],r['video'],r['local_query']):r for r in data['queries']}
    assert len(physical)==74 and len(query_rows)==3506
    signatures={};full_stats={};ranks=0
    for (target,video),record in physical.items():
        bank=metadata(target)['videos'][video];tracks=record['tracks'];times=defaultdict(set)
        for f,t in bank['observations']:times[int(t)].add(int(f))
        assert set(times)==set(tracks);first={t:min(v) for t,v in times.items()}
        delta={'KITTI-V2':1,'Dance':2,'SportsMOT':3}[target];assert delta==record['delta']
        for requested,old in record['views'].items():
            contexts={t:{(f,'CROP' if requested=='CLIP' and f-first[t]<delta else requested)
                         for f in fs} for t,fs in times.items()}
            graph,stats=graph_from_track_contexts(tracks,contexts)
            assert all(stats[k]==old[k] for k in stats)
            columns={t:i for i,t in enumerate(tracks)};matrix=np.zeros((len(old['forest']),len(tracks)))
            seen=set()
            for i,(f,v,a,b) in enumerate(old['forest']):
                assert (f,v) in contexts[a]&contexts[b] and graph.has_edge(a,b)
                matrix[i,columns[a]]=1;matrix[i,columns[b]]=-1;seen.update(((f,v,a),(f,v,b)))
            numeric=int(np.linalg.matrix_rank(matrix,tol=1e-10)) if len(matrix) else 0
            assert numeric==stats['quotient_rank'] and len(seen)==old['diagnostic_forest_unique_binary_requests']
            stats['diagnostic_forest_unique_binary_requests']=len(seen)
            full_stats[target,video,requested]=stats;ranks+=1
        p=ROOT/'results/runs/multidomain_generality_20261101'/KEYS[target]/'metadata.json'
        if str(p) not in signatures:signatures[str(p)]=file_sha(p)
    checked=0
    for pair,(_,target) in PAIRS.items():
        totals=Counter();full_components=[];library_components=[]
        for video in sorted(metadata(target)['videos']):
            record=physical[target,video];tracks=record['tracks'];first={int(k):v for k,v in record['first'].items()}
            actions,quotas=inventory(pair,video);by_query=defaultdict(lambda:defaultdict(set));requested={}
            for action in actions.values():
                q=action['local_query'];r=action['case'];f=int(r['frame_id']);t=int(r['track_id']);v=r['view']
                assert requested.setdefault(q,v)==v
                actual='CROP' if v=='CLIP' and f-first[t]<record['delta'] else v
                by_query[q][t].add((f,actual))
            for q,budget in quotas.items():
                _,stats=graph_from_track_contexts(tracks,by_query[q]);old=query_rows[pair,video,q]['library']
                assert all(stats[k]==old[k] for k in stats)
                entire=full_stats[target,video,requested[q]];n=len(tracks)
                totals.update(query_episodes=1,binary_request_budget=budget,candidate_query_tracks=n,
                    full_pairable_query_tracks=entire['pairable_tracks'],library_pairable_query_tracks=stats['pairable_tracks'],
                    full_no_peer_query_tracks=n-entire['pairable_tracks'],library_no_peer_query_tracks=n-stats['pairable_tracks'],
                    full_quotient_rank=entire['quotient_rank'],library_quotient_rank=stats['quotient_rank'],
                    queries_with_zero_library_comparisons=stats['quotient_rank']==0,
                    queries_with_connected_full_quotient=entire['quotient_components']==1,
                    budget_below_all_track_touch_lower_bound=budget<n,
                    budget_can_pay_full_diagnostic_forest=budget>=entire['diagnostic_forest_unique_binary_requests'])
                full_components.append(entire['quotient_components']);library_components.append(stats['quotient_components']);checked+=1
            for policy in ('DESIGN','COVERAGE_ONLY','PRIOR','UNIFORM'):
                p=PREP/pair/video/(policy+'_requests.json');signatures[str(p)]=file_sha(p)
        reproduced=dict(**totals,full_quotient_components_median=float(np.median(full_components)),
            library_quotient_components_median=float(np.median(library_components)))
        assert reproduced==primary['settings'][pair],pair
        print(json.dumps(dict(setting=pair,status='EXACT_ALL_METADATA_COUNTS')),flush=True)
    assert ranks==222 and checked==3506 and len(signatures)==595 and not guard
    local=SIX/'same_context_comparison_support_20261124/input_signatures.json';atomic_json(local,signatures)
    result=dict(status='PASS_INDEPENDENT_ALL_CONTEXT_GRAPH_SUPPORT_COUNTS_AND_NUMERIC_RANK',
        independent_algorithm='Track-context set intersections and NetworkX components; separate NumPy incidence matrix ranks.',
        full_video_view_graphs=ranks,library_query_graphs=checked,numeric_forest_ranks=ranks,
        input_files_hashed=len(signatures),local_input_signatures=str(local),input_signatures_sha256=file_sha(local),
        primary_report_sha256=file_sha(path),source_sha256=file_sha(Path(__file__)),
        manual_GT_open_attempts=0,no_new_teacher_GPU_fits_features_or_predictions=True,goal_status='ACTIVE')
    atomic_json(PUBLIC/'same_context_support_reproduction_20261124.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
