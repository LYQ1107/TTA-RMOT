# Positive-Free full complete-bank results

Development-target experiment; old Frozen HOTA 8.206 uses an incomplete bank and is historical only.

| Method | HOTA | DetA | AssA | DetRe | DetPr | IDF1 | IDSW | ΔHOTA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ZERO_UPDATE | 8.2180 | 3.1937 | 21.1470 | 4.5441 | 9.6868 | 4.4414 | 269.0000 | +0.0000 |
| RELATIVE_ONLY | 7.5705 | 4.1857 | 13.7120 | 11.1040 | 6.2855 | 5.5746 | 829.0000 | -0.6475 |
| PF_RANK | 8.2433 | 3.2336 | 21.0150 | 4.6120 | 9.7458 | 4.4947 | 269.0000 | +0.0253 |
| PF_RANK_GUARD | 8.2444 | 3.2342 | 21.0160 | 4.6134 | 9.7455 | 4.4956 | 269.0000 | +0.0264 |
| PF_IDENTITY | 8.8179 | 4.0132 | 19.3760 | 5.9354 | 11.0030 | 5.4476 | 347.0000 | +0.5999 |
| PF_HYBRID | 8.3213 | 3.4393 | 20.1330 | 4.9570 | 10.0800 | 4.7114 | 300.0000 | +0.1033 |

Completed methods: 6/6. No missing method is treated as complete.

Per-video metrics pool 17 sequences using official HOTA/Identity combine_sequences; they are not unweighted macro means.
