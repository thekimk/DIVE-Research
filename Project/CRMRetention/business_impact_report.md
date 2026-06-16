# Table 1. 시나리오별 비즈니스 임팩트 및 예측 성능 비교

| 시나리오 | AUUC | Top 10% Expected Uplift | Top 20% Expected Uplift | Top 30% Expected Uplift | Top 40% Expected Uplift | Top 50% Expected Uplift | Negative CATE Avoidance |
| ---- | ---- | ----------------------- | ----------------------- | ----------------------- | ----------------------- | ----------------------- | ----------------------- |
| Random Targeting (Baseline) | 0.500 | 782 | 1,564 | 2,346 | 3,128 | 3,910 | 0.0% |
| Baseline + T(Calendar) | 3.981 ± 0.431 | 1,820 | 1,935 | 1,986 | 1,975 | 1,964 | 7.3% |
| Baseline + T(Fatigue) | 0.530 ± 0.005 | 19,230 | 26,517 | 30,616 | 33,047 | 34,435 | 57.7% |
| Baseline + T(Risk) | 0.609 ± 0.003 | 20,792 | 26,799 | 29,612 | 30,957 | 31,650 | 42.5% |
| Baseline + T(Preference) | 0.504 ± 0.004 | 4,661 | 5,098 | 4,976 | 4,413 | 3,163 | 4.8% |