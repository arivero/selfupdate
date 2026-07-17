# v4 certification data appendix — agpuh01, 2026-07-17

Machine-extracted from runs/*/metrics.jsonl (this file is generated;
regenerate rather than hand-edit). Floor: train.v4_min_train_gpu_util=50
armed on the 4B/27B probes; 0.6B legs ran ungated (mechanics only).

## 0.6B v4 1proc e1

### single

| epoch | ev/s | seconds | util% | prep_s | exec_s | boundary_s |
|---|---|---|---|---|---|---|
| 1 | 34413 | 6.78 | None | — | — | — |

- teacher-forced e1: CE 0.1362 KL 0.0001 (8334 tok)
- locality: passed=True cross=0.0 vocab=0.0
- done: graceful=False vram_gb=13.24

## 0.6B v4 4stage e2

### stage0

| epoch | ev/s | seconds | util% | prep_s | exec_s | boundary_s |
|---|---|---|---|---|---|---|
| 1 | 12886 | 4.53 | None | — | — | — |
| 2 | 300610 | 0.19 | None | — | — | — |

- locality: passed=True cross=0.0 vocab=0.0
- done: graceful=False vram_gb=6.2

### stage1

| epoch | ev/s | seconds | util% | prep_s | exec_s | boundary_s |
|---|---|---|---|---|---|---|
| 1 | 12278 | 4.75 | None | — | — | — |
| 2 | 300296 | 0.19 | None | — | — | — |

- locality: passed=True cross=0.0 vocab=0.0
- done: graceful=False vram_gb=6.2

### stage2

| epoch | ev/s | seconds | util% | prep_s | exec_s | boundary_s |
|---|---|---|---|---|---|---|
| 1 | 11983 | 4.87 | None | — | — | — |
| 2 | 299751 | 0.19 | None | — | — | — |

- locality: passed=True cross=0.0 vocab=0.0
- done: graceful=False vram_gb=6.2

### stage3

| epoch | ev/s | seconds | util% | prep_s | exec_s | boundary_s |
|---|---|---|---|---|---|---|
| 1 | 12570 | 4.64 | None | — | — | — |
| 2 | 211265 | 0.28 | None | — | — | — |

- teacher-forced e1: CE 0.1362 KL 0.0001 (8334 tok)
- student e1: CE 2.6807 KL 2.4923
- teacher-forced e2: CE 0.1362 KL 0.0001 (8334 tok)
- student e2: CE 2.6807 KL 2.4922
- locality: passed=True cross=0.0 vocab=0.0
- done: graceful=False vram_gb=6.2

## 0.6B v4 3stage e2

### stage0

| epoch | ev/s | seconds | util% | prep_s | exec_s | boundary_s |
|---|---|---|---|---|---|---|
| 1 | 16549 | 4.53 | 28.7 | — | — | — |
| 2 | 302847 | 0.25 | 54.6 | — | — | — |

- locality: passed=True cross=0.0 vocab=0.0
- done: graceful=False vram_gb=6.91

### stage1

| epoch | ev/s | seconds | util% | prep_s | exec_s | boundary_s |
|---|---|---|---|---|---|---|
| 1 | 14355 | 5.23 | 27.2 | — | — | — |
| 2 | 298682 | 0.25 | 64.2 | — | — | — |

- locality: passed=True cross=0.0 vocab=0.0
- done: graceful=False vram_gb=6.91

### stage2

| epoch | ev/s | seconds | util% | prep_s | exec_s | boundary_s |
|---|---|---|---|---|---|---|
| 1 | 16621 | 5.01 | 29.6 | — | — | — |
| 2 | 230379 | 0.36 | 75.5 | — | — | — |

- teacher-forced e1: CE 0.1362 KL 0.0001 (8334 tok)
- student e1: CE 2.6807 KL 2.4923
- teacher-forced e2: CE 0.1362 KL 0.0001 (8334 tok)
- student e2: CE 2.6807 KL 2.4922
- locality: passed=True cross=0.0 vocab=0.0
- done: graceful=False vram_gb=7.27

## 0.6B adam+aligned e2

### single

| epoch | ev/s | seconds | util% | prep_s | exec_s | boundary_s |
|---|---|---|---|---|---|---|
| 1 | 58882 | 7.27 | None | — | — | — |
| 2 | 390363 | 1.10 | None | — | — | — |

- teacher-forced e1: CE 0.1362 KL 0.0001 (8334 tok)
- student e1: CE 2.6812 KL 2.4928
- teacher-forced e2: CE 0.1362 KL 0.0001 (8334 tok)
- student e2: CE 2.6813 KL 2.4927
- locality: passed=True cross=0.0 vocab=0.0
- done: graceful=False vram_gb=14.82

## 0.6B kv-refresh e2

### single

| epoch | ev/s | seconds | util% | prep_s | exec_s | boundary_s |
|---|---|---|---|---|---|---|
| 1 | 38599 | 6.05 | None | — | — | — |
| 2 | 95088 | 2.45 | None | — | — | — |

- teacher-forced e1: CE 0.1362 KL 0.0001 (8334 tok)
- student e1: CE 2.6807 KL 2.4923
- teacher-forced e2: CE 0.1362 KL 0.0001 (8334 tok)
- student e2: CE 2.6807 KL 2.4922
- locality: passed=True cross=0.0 vocab=0.0
- done: graceful=False vram_gb=13.66

## 4B PPP1 e12

### single

| epoch | ev/s | seconds | util% | prep_s | exec_s | boundary_s |
|---|---|---|---|---|---|---|
| 1 | 10053 | 14.47 | 77.8 | — | — | — |
| 2 | 17855 | 8.15 | 100.0 | — | — | — |
| 3 | 17833 | 8.16 | 99.9 | — | — | — |
| 4 | 17828 | 8.16 | 99.4 | — | — | — |
| 5 | 17857 | 8.14 | 99.8 | — | — | — |
| 6 | 17868 | 8.14 | 100.0 | — | — | — |
| 7 | 17829 | 8.16 | 100.0 | — | — | — |
| 8 | 17875 | 8.14 | 99.5 | — | — | — |
| 9 | 17867 | 8.14 | 100.0 | — | — | — |
| 10 | 17868 | 8.14 | 100.0 | — | — | — |
| 11 | 17875 | 8.14 | 100.0 | — | — | — |
| 12 | 17872 | 8.14 | 100.0 | — | — | — |

- teacher-forced e1: CE 0.0360 KL 0.0013 (4545 tok)
- student e1: CE 3.0403 KL 2.9679
- teacher-forced e2: CE 0.0360 KL 0.0013 (4545 tok)
- student e2: CE 3.0402 KL 2.9677
- teacher-forced e3: CE 0.0360 KL 0.0013 (4545 tok)
- student e3: CE 3.0402 KL 2.9678
- teacher-forced e4: CE 0.0360 KL 0.0013 (4545 tok)
- student e4: CE 3.0404 KL 2.9678
- teacher-forced e5: CE 0.0360 KL 0.0013 (4545 tok)
- student e5: CE 3.0401 KL 2.9676
- teacher-forced e6: CE 0.0360 KL 0.0013 (4545 tok)
- student e6: CE 3.0402 KL 2.9676
- teacher-forced e7: CE 0.0360 KL 0.0013 (4545 tok)
- student e7: CE 3.0408 KL 2.9683
- teacher-forced e8: CE 0.0360 KL 0.0013 (4545 tok)
- student e8: CE 3.0394 KL 2.9670
- teacher-forced e9: CE 0.0360 KL 0.0013 (4545 tok)
- student e9: CE 3.0400 KL 2.9675
- teacher-forced e10: CE 0.0360 KL 0.0013 (4545 tok)
- student e10: CE 3.0404 KL 2.9679
- teacher-forced e11: CE 0.0360 KL 0.0013 (4545 tok)
- student e11: CE 3.0395 KL 2.9671
- teacher-forced e12: CE 0.0360 KL 0.0013 (4545 tok)
- student e12: CE 3.0395 KL 2.9671

## 4B PPP2 e12

### stage0

| epoch | ev/s | seconds | util% | prep_s | exec_s | boundary_s |
|---|---|---|---|---|---|---|
| 1 | 8792 | 8.27 | 83.9 | — | — | — |
| 2 | 18047 | 4.03 | 99.9 | — | — | — |
| 3 | 18086 | 4.02 | 99.4 | — | — | — |
| 4 | 18086 | 4.02 | 99.8 | — | — | — |
| 5 | 18119 | 4.01 | 99.4 | — | — | — |
| 6 | 18073 | 4.02 | 99.1 | — | — | — |
| 7 | 18009 | 4.04 | 99.3 | — | — | — |
| 8 | 18078 | 4.02 | 98.8 | — | — | — |
| 9 | 18105 | 4.02 | 97.8 | — | — | — |
| 10 | 18116 | 4.01 | 100.0 | — | — | — |
| 11 | 18112 | 4.02 | 99.9 | — | — | — |
| 12 | 18114 | 4.01 | 93.4 | — | — | — |


### stage1

| epoch | ev/s | seconds | util% | prep_s | exec_s | boundary_s |
|---|---|---|---|---|---|---|
| 1 | 8401 | 8.66 | 76.6 | — | — | — |
| 2 | 17574 | 4.14 | 99.8 | — | — | — |
| 3 | 17571 | 4.14 | 98.0 | — | — | — |
| 4 | 17575 | 4.14 | 95.2 | — | — | — |
| 5 | 17589 | 4.13 | 98.2 | — | — | — |
| 6 | 17577 | 4.14 | 99.9 | — | — | — |
| 7 | 17579 | 4.14 | 99.9 | — | — | — |
| 8 | 17576 | 4.14 | 99.9 | — | — | — |
| 9 | 17591 | 4.13 | 99.9 | — | — | — |
| 10 | 17580 | 4.14 | 99.9 | — | — | — |
| 11 | 17606 | 4.13 | 96.2 | — | — | — |
| 12 | 17630 | 4.12 | 97.9 | — | — | — |

- teacher-forced e1: CE 0.0360 KL 0.0013 (4545 tok)
- student e1: CE 3.0403 KL 2.9679
- teacher-forced e2: CE 0.0360 KL 0.0013 (4545 tok)
- student e2: CE 3.0402 KL 2.9677
- teacher-forced e3: CE 0.0360 KL 0.0013 (4545 tok)
- student e3: CE 3.0402 KL 2.9678
- teacher-forced e4: CE 0.0360 KL 0.0013 (4545 tok)
- student e4: CE 3.0404 KL 2.9678
- teacher-forced e5: CE 0.0360 KL 0.0013 (4545 tok)
- student e5: CE 3.0401 KL 2.9676
- teacher-forced e6: CE 0.0360 KL 0.0013 (4545 tok)
- student e6: CE 3.0402 KL 2.9676
- teacher-forced e7: CE 0.0360 KL 0.0013 (4545 tok)
- student e7: CE 3.0408 KL 2.9683
- teacher-forced e8: CE 0.0360 KL 0.0013 (4545 tok)
- student e8: CE 3.0394 KL 2.9670
- teacher-forced e9: CE 0.0360 KL 0.0013 (4545 tok)
- student e9: CE 3.0402 KL 2.9677
- teacher-forced e10: CE 0.0360 KL 0.0013 (4545 tok)
- student e10: CE 3.0404 KL 2.9679
- teacher-forced e11: CE 0.0360 KL 0.0013 (4545 tok)
- student e11: CE 3.0394 KL 2.9670
- teacher-forced e12: CE 0.0360 KL 0.0013 (4545 tok)
- student e12: CE 3.0395 KL 2.9670

## 4B PPP3 e12

### stage0

| epoch | ev/s | seconds | util% | prep_s | exec_s | boundary_s |
|---|---|---|---|---|---|---|
| 1 | 7223 | 6.92 | 88.9 | 2.232 | 4.654 | 13.533 |
| 2 | 17130 | 2.92 | 99.8 | 0.011 | 2.904 | 11.183 |
| 3 | 16924 | 2.95 | 94.5 | 0.015 | 2.906 | 13.057 |
| 4 | 16890 | 2.96 | 99.4 | 0.01 | 2.903 | 60.226 |
| 5 | 17117 | 2.92 | 100.0 | 0.015 | 2.903 | 13.038 |
| 6 | 17053 | 2.93 | 99.5 | 0.01 | 2.905 | 13.043 |
| 7 | 17083 | 2.93 | 93.3 | 0.014 | 2.903 | 10.988 |
| 8 | 17165 | 2.91 | 100.0 | 0.009 | 2.901 | 62.19 |
| 9 | 17012 | 2.94 | 100.0 | 0.012 | 2.904 | 10.991 |
| 10 | 17107 | 2.92 | 99.9 | 0.015 | 2.905 | 11.024 |
| 11 | 16962 | 2.95 | 96.1 | 0.01 | 2.904 | 11.154 |
| 12 | 17072 | 2.93 | 99.9 | 0.01 | 2.905 | 53.706 |

- locality: passed=True cross=0.0 vocab=0.0
- done: graceful=False vram_gb=41.1

### stage1

| epoch | ev/s | seconds | util% | prep_s | exec_s | boundary_s |
|---|---|---|---|---|---|---|
| 1 | 7257 | 6.89 | 75.7 | 2.424 | 4.457 | 245.85 |
| 2 | 18373 | 2.72 | 90.4 | 0.014 | 2.698 | 13.057 |
| 3 | 18373 | 2.72 | 90.8 | 0.016 | 2.702 | 11.036 |
| 4 | 18348 | 2.72 | 90.7 | 0.015 | 2.698 | 13.079 |
| 5 | 18290 | 2.73 | 88.9 | 0.015 | 2.699 | 61.176 |
| 6 | 18386 | 2.72 | 90.9 | 0.015 | 2.701 | 13.037 |
| 7 | 18381 | 2.72 | 97.2 | 0.015 | 2.699 | 13.216 |
| 8 | 18310 | 2.73 | 89.6 | 0.014 | 2.7 | 11.128 |
| 9 | 18347 | 2.72 | 90.5 | 0.018 | 2.699 | 61.28 |
| 10 | 18039 | 2.77 | 88.5 | 0.021 | 2.7 | 11.08 |
| 11 | 18309 | 2.73 | 90.3 | 0.014 | 2.699 | 11.178 |
| 12 | 18093 | 2.76 | 91.5 | 0.015 | 2.7 | 11.056 |

- locality: passed=True cross=0.0 vocab=0.0
- done: graceful=False vram_gb=41.17

### stage2

| epoch | ev/s | seconds | util% | prep_s | exec_s | boundary_s |
|---|---|---|---|---|---|---|
| 1 | 6912 | 6.58 | 70.4 | 2.114 | 4.335 | 245.316 |
| 2 | 18008 | 2.52 | 100.0 | 0.014 | 2.4 | 12.495 |
| 3 | 17996 | 2.53 | 100.0 | 0.014 | 2.402 | 12.505 |
| 4 | 17726 | 2.56 | 97.2 | 0.017 | 2.402 | 12.497 |
| 5 | 17840 | 2.55 | 97.3 | 0.017 | 2.401 | 62.57 |
| 6 | 17804 | 2.55 | 99.9 | 0.02 | 2.403 | 12.551 |
| 7 | 18011 | 2.52 | 97.2 | 0.013 | 2.4 | 12.539 |
| 8 | 18015 | 2.52 | 99.0 | 0.014 | 2.399 | 12.525 |
| 9 | 17912 | 2.54 | 99.8 | 0.022 | 2.404 | 60.545 |
| 10 | 17853 | 2.55 | 99.8 | 0.016 | 2.401 | 12.522 |
| 11 | 17978 | 2.53 | 99.8 | 0.016 | 2.401 | 10.512 |
| 12 | 18002 | 2.52 | 100.0 | 0.014 | 2.401 | 10.512 |

- teacher-forced e1: CE 0.0360 KL 0.0013 (4545 tok)
- student e1: CE 3.0403 KL 2.9679
- teacher-forced e2: CE 0.0360 KL 0.0013 (4545 tok)
- student e2: CE 3.0402 KL 2.9677
- teacher-forced e3: CE 0.0360 KL 0.0013 (4545 tok)
- student e3: CE 3.0402 KL 2.9678
- teacher-forced e4: CE 0.0360 KL 0.0013 (4545 tok)
- student e4: CE 3.0404 KL 2.9678
- teacher-forced e5: CE 0.0360 KL 0.0013 (4545 tok)
- student e5: CE 3.0401 KL 2.9676
- teacher-forced e6: CE 0.0360 KL 0.0013 (4545 tok)
- student e6: CE 3.0402 KL 2.9676
- teacher-forced e7: CE 0.0360 KL 0.0013 (4545 tok)
- student e7: CE 3.0408 KL 2.9683
- teacher-forced e8: CE 0.0360 KL 0.0013 (4545 tok)
- student e8: CE 3.0394 KL 2.9670
- teacher-forced e9: CE 0.0360 KL 0.0013 (4545 tok)
- student e9: CE 3.0402 KL 2.9677
- teacher-forced e10: CE 0.0360 KL 0.0013 (4545 tok)
- student e10: CE 3.0403 KL 2.9679
- teacher-forced e11: CE 0.0360 KL 0.0013 (4545 tok)
- student e11: CE 3.0395 KL 2.9670
- teacher-forced e12: CE 0.0360 KL 0.0013 (4545 tok)
- student e12: CE 3.0395 KL 2.9670
- locality: passed=True cross=0.0 vocab=0.0
- done: graceful=False vram_gb=40.42

## 27B PPP4 e12

### stage0

| epoch | ev/s | seconds | util% | prep_s | exec_s | boundary_s |
|---|---|---|---|---|---|---|
| 1 | 3568 | 15.17 | 57.9 | 4.926 | 10.175 | 6.59 |
| 2 | 18231 | 2.97 | 91.8 | 0.011 | 2.929 | 3.095 |
| 3 | 18011 | 3.00 | 98.1 | 0.01 | 2.924 | 3.111 |
| 4 | 18030 | 3.00 | 91.1 | 0.01 | 2.925 | 134.455 |
| 5 | 18298 | 2.96 | 97.8 | 0.01 | 2.927 | 3.267 |
| 6 | 18232 | 2.97 | 93.0 | 0.012 | 2.945 | 3.193 |
| 7 | 18171 | 2.98 | 94.1 | 0.01 | 2.929 | 3.154 |
| 8 | 18226 | 2.97 | 92.9 | 0.01 | 2.93 | 135.26 |
| 9 | 18149 | 2.98 | 95.7 | 0.01 | 2.93 | 3.163 |
| 10 | 18101 | 2.99 | 93.1 | 0.009 | 2.927 | 3.196 |
| 11 | 18191 | 2.97 | 94.1 | 0.009 | 2.925 | 3.152 |
| 12 | 18243 | 2.97 | 94.4 | 0.009 | 2.924 | 140.159 |

- locality: passed=True cross=0.0 vocab=0.0
- done: graceful=False vram_gb=67.48

### stage1

| epoch | ev/s | seconds | util% | prep_s | exec_s | boundary_s |
|---|---|---|---|---|---|---|
| 1 | 3514 | 15.40 | 54.2 | 4.902 | 10.46 | 0.069 |
| 2 | 18309 | 2.96 | 94.8 | 0.011 | 2.932 | 0.076 |
| 3 | 18394 | 2.94 | 95.0 | 0.01 | 2.919 | 0.064 |
| 4 | 18287 | 2.96 | 97.8 | 0.011 | 2.923 | 0.076 |
| 5 | 18255 | 2.96 | 95.2 | 0.01 | 2.924 | 0.064 |
| 6 | 18278 | 2.96 | 92.6 | 0.01 | 2.921 | 0.11 |
| 7 | 17105 | 3.16 | 91.8 | 0.01 | 2.924 | 0.069 |
| 8 | 18380 | 2.94 | 96.8 | 0.01 | 2.924 | 0.065 |
| 9 | 17717 | 3.05 | 90.6 | 0.01 | 2.925 | 0.07 |
| 10 | 18344 | 2.95 | 94.1 | 0.01 | 2.926 | 0.065 |
| 11 | 18274 | 2.96 | 93.9 | 0.01 | 2.925 | 0.079 |
| 12 | 18289 | 2.96 | 96.5 | 0.01 | 2.925 | 0.065 |


### stage2

| epoch | ev/s | seconds | util% | prep_s | exec_s | boundary_s |
|---|---|---|---|---|---|---|
| 1 | 3559 | 15.20 | 62.9 | 4.818 | 10.367 | 0.068 |
| 2 | 18096 | 2.99 | 92.1 | 0.011 | 2.92 | 0.247 |
| 3 | 18390 | 2.94 | 94.2 | 0.01 | 2.922 | 0.068 |
| 4 | 18295 | 2.96 | 96.4 | 0.01 | 2.921 | 0.077 |
| 5 | 18179 | 2.98 | 94.2 | 0.01 | 2.922 | 0.07 |
| 6 | 16499 | 3.28 | 88.9 | 0.01 | 2.921 | 0.112 |
| 7 | 18259 | 2.96 | 92.9 | 0.01 | 2.918 | 0.064 |
| 8 | 18248 | 2.97 | 95.9 | 0.01 | 2.921 | 0.113 |
| 9 | 18375 | 2.94 | 94.8 | 0.01 | 2.922 | 0.065 |
| 10 | 18213 | 2.97 | 94.2 | 0.01 | 2.923 | 0.07 |
| 11 | 18164 | 2.98 | 92.1 | 0.01 | 2.922 | 0.074 |
| 12 | 18256 | 2.96 | 95.1 | 0.01 | 2.921 | 0.064 |


### stage3

| epoch | ev/s | seconds | util% | prep_s | exec_s | boundary_s |
|---|---|---|---|---|---|---|
| 1 | 3395 | 15.94 | 52.0 | 4.917 | 10.82 | 0.097 |
| 2 | 17437 | 3.10 | 95.0 | 0.055 | 2.936 | 0.077 |
| 3 | 17419 | 3.11 | 93.8 | 0.119 | 2.937 | 0.092 |
| 4 | 15791 | 3.43 | 82.8 | 0.055 | 2.938 | 0.095 |
| 5 | 17224 | 3.14 | 94.5 | 0.056 | 2.949 | 0.082 |
| 6 | 17226 | 3.14 | 96.6 | 0.056 | 2.947 | 0.065 |
| 7 | 17476 | 3.10 | 93.7 | 0.109 | 2.944 | 0.065 |
| 8 | 17535 | 3.09 | 95.1 | 0.11 | 2.938 | 0.075 |
| 9 | 17294 | 3.13 | 93.1 | 0.111 | 2.938 | 0.067 |
| 10 | 17425 | 3.11 | 96.1 | 0.055 | 2.939 | 0.066 |
| 11 | 17559 | 3.08 | 97.6 | 0.11 | 2.935 | 0.072 |
| 12 | 17239 | 3.14 | 94.5 | 0.119 | 2.935 | 0.07 |

- teacher-forced e1: CE 0.0185 KL 0.0004 (3382 tok)
- teacher-forced e2: CE 0.0185 KL 0.0004 (3382 tok)
- teacher-forced e3: CE 0.0185 KL 0.0004 (3382 tok)
- teacher-forced e4: CE 0.0185 KL 0.0004 (3382 tok)
- teacher-forced e5: CE 0.0185 KL 0.0004 (3382 tok)
- teacher-forced e6: CE 0.0185 KL 0.0004 (3382 tok)
- teacher-forced e7: CE 0.0185 KL 0.0004 (3382 tok)
- teacher-forced e8: CE 0.0185 KL 0.0004 (3382 tok)
- teacher-forced e9: CE 0.0185 KL 0.0004 (3382 tok)
- teacher-forced e10: CE 0.0185 KL 0.0004 (3382 tok)
- teacher-forced e11: CE 0.0185 KL 0.0004 (3382 tok)
- teacher-forced e12: CE 0.0185 KL 0.0004 (3382 tok)

## 27B stage-0 battery lines (from the stage log)

- epoch0 + after_epoch_4/8/12: standard 0.646 (delta +0.000, worst arc_easy +0.000)
- recall: machado 0.08-0.09, quijote_ch1 0.24-0.26, quijote_ch4 0.15
  (note: 27B epoch-zero baselines are far above 0.6B's 0.07/0.09/0.12)

## Incident log (certification day)

1. B>1 in-place mask bug in run_block (latent in-tree) - fixed.
2. Locality cert crashed on linear-attention layers post-training - fixed;
   4B PPP2 predates the fix (telemetry valid, certs absent by design).
3. 27B first launch OOM: auto-residency under-counted layer_major
   accumulation + B=100 linear-kernel intermediates - fixed (B=32,
   honest accounting); stale stage 0 needed SIGKILL (battery does not poll
   the cooperative stop - known gap).
4. Watcher double-launch near-miss - launch lease added.
5. Reaper false positive on stale log tracebacks killed the 27B siblings
   mid-drain after stage 0 finished cleanly - fixed (run-complete
   exemption); sibling training telemetry complete, their locality certs
   absent from this run.

