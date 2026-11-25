"""Truthful first-round report; a completed run is not a completed research goal."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.a100_visual_classical_prepare import OUT


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data=json.loads((OUT/'analysis/official_results.json').read_text())
    coverage=json.loads((OUT/'analysis/coverage.json').read_text())
    costs=json.loads((OUT/'analysis/cost_ledger.json').read_text())
    figures=OUT/'figures';figures.mkdir(parents=True,exist_ok=True)
    selected=['DIRECT_NATIVE','KD_HARD','KD_SOFT','KD_SOFT_4000','FRAME_KD_HARD','FRAME_KD_SOFT','FRAME_KD_SOFT_STRUCTURED','FULL','NO_CORRECTION','NO_GRAPH']
    fig,axes=plt.subplots(1,2,figsize=(14,6),sharex=True)
    for ax,pair in zip(axes,('C4','C1')):
        rows=data[pair]['methods'];values=[rows[('convergence/' if m.endswith('_4000') else 'predictions/')+'COVER__'+m]['HOTA'] for m in selected]
        colors=['#267c70' if m=='FULL' else '#9ca3af' if m.startswith('NO_') else '#5079a8' for m in selected]
        ax.barh(np.arange(len(values)),values,color=colors)
        ax.set_yticks(np.arange(len(values)),selected);ax.invert_yaxis();ax.set_xlabel('Official HOTA')
        ax.set_title(pair+' | same COVER evidence, seed 0')
        for y,value in enumerate(values):ax.text(value+.2,y,f'{value:.3f}',va='center',fontsize=8)
        ax.set_xlim(0,max(85,max(values)+7));ax.grid(axis='x',alpha=.15)
    fig.suptitle('Batch-transductive development diagnostics | frozen source representations | seed0')
    fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(figures/('same_evidence_controls.'+suffix),dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,pair in zip(axes,('C4','C1')):
        methods=('PRIOR','UNIFORM','UNCERTAINTY','COVER')
        values=[data[pair]['methods']['predictions/'+m+'__FULL']['HOTA'] for m in methods]
        ax.bar(methods,values,color=['#9ca3af','#9ca3af','#9ca3af','#267c70'])
        ax.set_title(pair+' | same inference and teacher budget');ax.set_ylabel('Official HOTA')
        ax.tick_params(axis='x',rotation=20);ax.set_ylim(0,max(85,max(values)+7))
        for i,value in enumerate(values):ax.text(i,value+.5,f'{value:.3f}',ha='center')
    fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(figures/('acquisition_controls.'+suffix),dpi=180)
    plt.close(fig)
    text=['跨域RMOT：视觉覆盖与可纠错经典推断——第一轮完整记录','',
        '状态：首轮机制实验，不自动宣称持续Goal已完成。',
        '设定：源模型冻结；目标域图像、查询、候选轨迹及冻结Qwen反馈用于适配。',
        '目标GT仅在所有方法封存后由独立评估程序读取。数据均已参与既有研发，非独立盲测。',
        '本轮seed0；后续重复与真实冷启动成本仍需完成。','',
        '信息范围更正：本轮KD在整批目标教师回答上拟合后回测，属于数据集转导。',
        '不能将其写成逐视频TTA；图方法本身只用本视频回答，与整批KD的对比属机制诊断。',
        '另有strict_prefix / exemplar_prefix逐视频因果对照；当前视频内仍允许转导，非逐帧因果。','',
        '方法：先验分层/视觉最远点覆盖；教师软概率锚、弱源先验与图拉普拉斯谐函数。',
        '图只用冻结PCD。固定超参数未扫描目标GT。输出分数不冒充已校准概率。',
        '教师基线包含硬/软直接预测、原生学生回退、实际400步轨迹均值与逐帧KD头，另有4000步轨迹KD收敛对照。',
        '逐帧KD使用教师实际看到的帧训练，再预测全部帧；另保留共同解码规则版本。',
        '这些KD均为冻结PCD上的真实残差MLP训练，不能写成完整骨干/LoRA蒸馏已被普遍击败。','']
    for pair in ('C4','C1'):
        result=data[pair];text.extend([pair,'方法 / HOTA / 预测行数'])
        for key,row in result['methods'].items():text.append(f"{key}: {row['HOTA']:.8f}; rows={row['prediction_rows']}")
        text.extend(['','预登记对照与原视频聚类描述性95%区间：'])
        for key,row in result['comparisons'].items():
            c=row['original_parent_video'];text.append(f"{key}: ΔHOTA={row['delta_HOTA']:.8f}; n={c['clusters']}; CI95={c['CI95']}")
        text.extend(['','覆盖账单（IoU匹配的候选观测，非可相加HOTA损失）：'])
        for key,row in coverage[pair]['acquisition'].items():text.append(key+': '+json.dumps(row,ensure_ascii=False))
        for key,row in coverage[pair]['outputs'].items():text.append(key+': '+json.dumps(row,ensure_ascii=False))
        c=costs[pair]['evidence_preparation'];text.extend(['','共享证据准备成本：',
            f"responses={c['responses']}; actual_new_calls={c['teacher']['new_calls']}; cache_hits={c['teacher']['cache_hits']}",
            f"teacher_load={c['teacher_load_wall_seconds']:.3f}s; service={c['teacher_service_wall_seconds']:.3f}s; terminal_seal={c['final_weight_seal_wall_seconds']:.3f}s",
            'worker process_wall从collection模块导入开始，包含此前输入准备；不可再与输入准备wall相加。',
            '历史PCD提取尚未在本轮重跑；共享回答池不是任一方法的真实部署墙钟。',''])
    text.extend(['研究解释边界：',
        '只有同问答控制才能讨论证据利用；不同选题结果同时改变可见监督位置。',
        '数量匹配控制用于区分排序和输出数量，计数来自方法自身预测，未使用GT数量。',
        '13个Sports原视频而不是45剪辑作为主重采样单位；Dance25个视频分别聚类。',
        '图传播可能传播教师错误；分层覆盖不能保证有限预算下找到所有真实目标。',
        '后续若修改依据本轮开发诊断，必须保留版本和所有负结果，不包装成盲测。'])
    (OUT/'first_round_report.txt').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(status='FIRST_ROUND_REPORT_WRITTEN_NOT_GOAL_COMPLETION',path=str(OUT/'first_round_report.txt'))),flush=True)


if __name__=='__main__':main()
