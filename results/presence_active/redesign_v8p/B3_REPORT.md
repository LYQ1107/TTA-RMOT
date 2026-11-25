## B3

| pair | method | rho | HOTA | DetA | AssA | IDF1 | delta_vs_MINUS | CI95_vs_MINUS | logical_calls | new_calls | cache_hits | reused |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C4 | FULL | 0.3 | 60.03685316632962 | 51.539 | 69.946 | 58.8 | 11.189775762973767 | [7.666913529862094, 15.131208188416359] | 3508 | 391 | 3117 | False |
| C4 | SLOW_REPLAY | 0.3 | 52.14626459554812 | 39.702 | 68.501 | 48.374 | 3.2991871921922638 | [0.3185326036661069, 6.660413690461902] | 3508 | 0 | 0 | False |

### 成本

| pair | method | rho | teacher_GPU_hours | original_reused_method_teacher_GPU_hours | teacher_equivalent_GPU_hours | LoRA_GPU_hours | rescore_GPU_hours | wall_seconds | GT_open_attempts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C4 | FULL | 0.3 | 0.006626883870145927 | NA | 0.08306926543012974 | 1.4311938471586392 | 0.8967535315759273 | 8550.20572760189 | 0 |
| C4 | SLOW_REPLAY | 0.3 | 0.0 | NA | 0.09629169495102868 | 0.5217034156937411 | 0.8377168126950144 | 4957.01973392209 | 0 |

复用行的teacher_GPU_hours=0只指本阶段没有新增forward，不是该系统无需GPU；原方法实际成本在original_reused_method_teacher_GPU_hours，缓存等价forward成本另列。

