"""Publish completed phase0 tables without reading runtime GT or changing inputs."""
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha


def table(rows,columns):
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']+
        ['| '+' | '.join(str(r.get(k,'NA')) for k in columns)+' |' for r in rows])


def report():
    public=ROOT/'results/presence_active/redesign_v8';folder=public/'amendments_A2_A3'
    read=lambda name:json.loads((folder/name).read_text())
    tests=read('cpu_tests.json');exact=read('stage0_v7_reproduction.json');em=read('stage0_offline_em.json')
    gt=read('stage0_GT_diagnostics.json');assert exact['status']=='PASS' and em['status']=='PASS' and gt['status']=='COMPLETE'
    estimates=list(csv.DictReader((folder/'stage0_estimators.csv').open()))
    decisions=[read('stage0_presence_'+p+'.json') for p in ('C1','C2','C3','C4','C5','C6')]
    kitti=list(csv.DictReader((folder/'stage0_kitti_pi_counterfactual.csv').open()))
    rel=list(csv.DictReader((folder/'stage0_reliability_vs_GT.csv').open()))
    synth=[]
    for rho in (.1,1.):
        value=read(tests.get('synthetic_result_subdirectory','')+f'/stage0_synthetic_rho_{rho}.json' if tests.get('synthetic_result_subdirectory') else f'stage0_synthetic_rho_{rho}.json')
        assert value['status']=='PASS'
        synth.append(dict(true_rho=rho,rho_hat=value['EM']['rho_hat'],pi_hat=value['EM']['pi_strata'],
            answered_tracks=value['EM']['tracks_per_stratum'],identification=value['EM']['identification_status'],
            maximum_iterations=max(x['after']['iterations'] for x in value['stream']),
            nonmonotone_fits=sum(not x['after']['objective_trace_monotone'] for x in value['stream']),
            old_pi0=value['old_v7_pi0_grouped_soft_counts']))
    cost=json.loads((public/'budget_preflight.json').read_text())
    preflight=json.loads((public/'frozen_input_preflight.json').read_text())
    text='# v8：A2/A3 阶段0完成，尚无新的正式RMOT结果\n\n'
    text+='最新状态：CPU验收和0b–0e完成；按原计划进入C1/C2前2视频冒烟。没有因下面的GT事后诊断改参数或新增门槛。A2/A3先公开于c22d9fbda8d1a71e1a85163cf3ff29acf6d52d41，再产生本轮结果。\n\n'
    text+='## 1. CPU验收与修订\n\n'
    text+=f"完整CPU套件：{tests['passed']} passed，{tests['failed']} failed；{tests['pytest_seconds']}秒。A1的rho=1边界、rho=.1误差<.05、可识别档pi误差<.07、旧v7 pi0偏差>.1均按注册规则执行。\n\n"
    text+=f"A2仅将iters改为5000，tol仍1e-7；每次fit记录不截断观测对数后验（含Beta先验）、1e-10单调性记录和最后两个delta的比值。单调性只是诊断：当前合规调度rho=.1有{synth[0]['nonmonotone_fits']}次fit出现下降，不改写成全程单调，也不据此新增停止条件。截断E步不等同于无截断目标的精确EM。初次A2/A3重放的109次下降及原始结果也保留。\n\n"
    text+='A3仅在每档已回答轨迹对>=200时验收pi；不足标UNIDENTIFIED_INSUFFICIENT_DATA。此标记是注册的样本量门槛，不证明模型正确或参数具有一般可识别性。存在性判定项计算次数和EM迭代项计算次数分别记录。\n\n'
    text+=table(synth,['true_rho','rho_hat','pi_hat','answered_tracks','identification','maximum_iterations','nonmonotone_fits','old_pi0'])+'\n\n'
    text+='rho=1的.95是截断区间的设计上限，不是估计误差；第2档只问3对、参与14次存在性判定项计算，因此不接受其.25619作为恢复真值的证明。初次同5122e3a调度的rho=.1为.10297748、pi=(.80239465,.30663253,.05)，原文件保留。代码检查发现GLANCE后可能进入low_tentative，预生成IDENTITY列表需在实际调度时再次过滤；此修复只落实原注册规则，未用GT调整参数。修复后的合成流在sampler_conformance目录，点值有所变化，但原rho/pi精度和旧v7回归条件仍全部通过。\n\n'
    text+='## 2. 0b：精确v7复现与GT-free EM重放\n\n'
    text+=f"v7六方向{exact['videos']}视频逐值复现最大误差{exact['max_absolute_error']}（要求<1e-9）。v8重放{len(em['records'])}视频全部收敛，最多{max(x['after']['iterations'] for x in em['records'])}次；GT打开尝试0、GPU/教师调用0。GT-free输出先封存，独立GT进程再读取。\n\n"
    text+='## 3. 0c：v7 / v8 EM / GT（全部为离线诊断）\n\n'
    text+=table(estimates,['pair','stratum','v7_rho','v7_global_pi','v8_rho','v8_pi','GT_rho','GT_pi_answered','GT_pi_all_eligible','answered_track_pairs','identification_status','presence_likelihood_evaluations'])+'\n\n'
    text+='18档均达到>=200，但真实数据EM的pi仍明显高估，不能因为合成恢复通过便声称已消除真实偏差。GT_pi_answered只统计有目标句中已回答轨迹，GT_pi_all_eligible统计有目标句全部合格轨迹；两者均按IoU>=.5一对一观测匹配后的轨迹多数>=.5定义。\n\n'
    text+='## 4. 0d：存在性反事实（不是新HOTA结果）\n\n'
    text+=table(decisions,['pair','present','empty','old_false_absent','old_true_absent','v8_false_absent','v8_true_absent','v8_false_absent_output','v8_true_absent_output','low_tentative'])+'\n\n'
    text+='反事实复用旧问题，没有补造GLANCE；使用保存的初始档位、v7最终弱排名和原逐轮DS。本表是近似离线归因，不是闭环v8结果。output列还计PRIOR_BUDGET_EXHAUSTED；evidence列不把待定句当已证实为空。C2/C5误判空反而增加，这一负面结果全部保留；原阶段0没有该项数值门槛，按已注册计划继续真实冒烟而不调参。\n\n'
    text+='## 5. 0e：KITTI专项（ORACLE，仅归因）\n\n'
    text+=table(kitti,['pair','original_false_absent','flipped_with_GT_pi_answered','flipped_with_GT_pi_all_eligible'])+'\n\n'
    text+='只在独立诊断进程以GT pi替换原v7逐答公式中的pi，固定原rho、DS与弱排名，重放原有回答、不产生新问题；C2的296句和C5的12句均不再触发判空。这是固定问答下的局部ORACLE反事实，不是允许在线读取GT，也不能声称真实v8已恢复这些句子或对应GT轨迹。逐答档位、看法、GT、原pi、EM视频初pi、原LLR见stage0_kitti_answer_decomposition.csv（3142行）。\n\n'
    text+=table([r for r in rel if r['view']=='FRAME_BOX'],['pair','view','DS_specificity','GT_specificity','DS_minus_GT'])+'\n\n'
    text+='C2 FRAME_BOX特异度DS=.759826、GT=.467442，偏差.292384；按指令只报告、不修复。GT值来自所问回答行的轨迹多数标签，不能当作无偏总体估计。\n\n'
    text+='## 6. 下一阶段准入、成本和保护\n\n'
    text+=f"六方向首视频PRES_TOPK逐行一致检查通过（{preflight['status']}），没有加载FlexHook/PCD。11项固定阶段2/3的保守预算{cost['total_GPU_hours_estimate']} GPU小时，未扣缓存、跳过或已完成冒烟；不是实际成本或新增授权范围。全部新帧缓存估计{cost['frame_cache_bytes_estimate']}字节，data1准入可用{cost['data1_available_bytes']}字节。沿已授权新缓存data1+data2指定运行路径链接；历史资产不迁移、不删除。最多同时2张共享卡，入场显存足够时优先少进程。\n\n"
    text+='本阶段真实教师/学生调用与GPU时间均为0；CPU套件用时及离线EM232.654322秒分别记录，不相加冒充GPU成本。第一次preflight调用漏设v8 manifest，被原授权检查立即拒绝、未写旧结果；补齐显式环境变量后只重跑该预检通过，不绕过gate。\n\n'
    text+='原200迭代失败的JSON、测试记录和停止报告仍在历史提交5122e3a及原路径；失败JSON SHA9b4315cf3c40ff3f8fd037d12aa2480aba967d424f070e34174693303848e270保持不变。v7所有代码、结果不改。大数值状态留服务器，合并JSON保留每个路径/SHA；表格、曲线、源代码、收据均提交。\n\n'
    text+='下一步仅C1/C2 MINUS前2视频；通过后恢复同一视频流继续六方向MINUS，并运行FULL C1/C4/C6、固定存在性REPLAY C1/C4。没有启动旧80项、v6扩展或阶段4。\n'
    (folder/'REPORT.md').write_text(text)
    (public/'REPORT.md').write_text(text)
    atomic_json(folder/'stage0_complete.json',dict(status='COMPLETE_STAGE0_A2_A3',
        exact_v7_error=exact['max_absolute_error'],real_replay_videos=len(em['records']),
        all_real_EM_converged=True,CPU_tests=tests,GT_feedback_to_runtime=False,
        generation_GT_open_attempts=0,GPU_calls=0,teacher_calls=0,
        prereg_sha256=file_sha(public/'PREREG.md'),manifest_sha256=file_sha(public/'manifest.json')))
    print(json.dumps(dict(status='COMPLETE_STAGE0_A2_A3',report=str(public/'REPORT.md'))))


if __name__=='__main__':report()
