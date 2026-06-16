# Causal Forest DML 비즈니스 평가 지표 및 코드 원리

이 문서는 [evaluate_ablation_business_impact.py](file:///c:/Users/user/Desktop/casual%20ml/scratch/evaluate_ablation_business_impact.py)에 구현된 4가지 인과추론 비즈니스 평가 핵심 함수의 연산 원리와 수식을 정리한 가이드라인입니다.

---

## 1. `compute_gain_curve` (누적 마케팅 효과 곡선 계산)
> **역할:** 고객을 한 명씩 추가해가며 마케팅 누적 효과를 누적합으로 쌓아 올리는 함수입니다.

### 💻 코드 구현
```python
def compute_gain_curve(y, t, cate):
    # 1. 효과가 큰 고객 순서대로 정렬
    order = np.argsort(cate)[::-1]
    y_sorted = y[order]
    t_sorted = t[order]
    
    # 2. 누적 고객 수(n_t, n_c) 및 누적 구매수(y_t, y_c) 계산
    n_t = np.cumsum(t_sorted == 1)
    n_c = np.cumsum(t_sorted == 0)
    y_t = np.cumsum(y_sorted * (t_sorted == 1))
    y_c = np.cumsum(y_sorted * (t_sorted == 0))
    
    # 3. 각 누적 지점에서의 구매율 차이 계산 후 누적 고객수 곱하기
    mean_y_t = y_t / n_t
    mean_y_c = y_c / n_c
    gain = (mean_y_t - mean_y_c) * (n_t + n_c)
    
    return gain
```

### 🔍 원리 설명
* **정렬 (Sorting):** 모델이 예측한 마케팅 효과($\hat{\tau}_i$, CATE)가 가장 높은 고객부터 1등부터 꼴찌까지 내림차순으로 줄을 세웁니다.
* **누적 집계 (Cumulative Sum):** 상위 1명부터 전체 고객까지 범위를 순차적으로 늘려가면서, 해당 범위 내 **광고를 받은 사람 수 ($N_t$)**, **안 받은 사람 수 ($N_c$)**, 그리고 각각의 **실제 구매 건수 ($Y_t, Y_c$)**를 누적으로 더해나갑니다.
* **효과 계산 (Gain Estimation):** 각 누적 범위마다 아래 공식을 적용하여 **마케팅 덕분에 순수하게 늘어난 추가 구매량(Gain)**을 한 단계씩 기록합니다.
  $$\text{Gain}(k) = \left( \frac{Y_t(k)}{N_t(k)} - \frac{Y_c(k)}{N_c(k)} \right) \times \left( N_t(k) + N_c(k) \right)$$

---

## 2. `calculate_auuc_bootstrap` (부트스트랩 기반 면적 계산)
> **역할:** 누적 마케팅 효과 곡선 아래의 부호가 포함된 면적(AUUC)을 구하고, 실험의 통계적 신뢰도(오차 범위 $\pm$)를 계산합니다.

### 💻 코드 구현
```python
def calculate_auuc_bootstrap(y, t, cate, n_bootstraps=50, seed=42):
    # 1. 오리지널 테스트셋의 최종 마케팅 효과(최종 Gain)를 구하여 기준으로 삼음
    final_gain_orig = (y_t_orig / n_t_orig - y_c_orig / n_c_orig) * n
    abs_final_gain_orig = abs(final_gain_orig)
    
    # 2. 부트스트랩 (50번 복원 추출 복제 실행)
    for _ in range(n_bootstraps):
        boot_idx = rng.choice(n, size=n, replace=True)
        # ... (샘플링 후 gain 계산) ...
        
        # 3. 곡선 면적 구하기 (평균값 계산을 통한 적분)
        normalized_gain = gain / abs_final_gain_orig
        auuc = np.mean(normalized_gain)
```

### 🔍 원리 설명
* **기준점 설정 (Normalization):** 전체 집단에 광고를 다 보냈을 때 얻는 최종 효과(`abs_final_gain_orig`)를 분모로 두어, 스케일이 서로 다른 여러 시나리오 그래프를 동일 선상에서 평가할 수 있도록 정규화합니다.
* **부트스트랩 (Bootstrap):** 단 한번의 면적 계산은 데이터 노이즈에 왜곡될 우려가 있습니다. 따라서 데이터를 무작위 복원추출로 50번 복제해서 매번 면적을 계산하고, 이들의 평균값과 표준오차(예: `0.530 ± 0.005`)를 산출합니다.
* **부호 포함 면적 적분 (Signed Integral):** 수학적으로 정규화된 그래프 높이 값들의 평균(`np.mean`)을 구하는 것은 **0기준선 기반의 부호를 포함한 면적(AUUC)**을 구하는 적분 연산과 완전히 동일합니다. (위쪽은 $+$, 아래쪽은 $-$로 상쇄 누적됨)

---

## 3. `calculate_expected_uplift` (예상 추가 구매 수 계산)
> **역할:** 마케팅을 실행했을 때 실제 늘어날 판매량(구매 건수)의 절대 수치를 구하는 직관적인 비즈니스 평가지표입니다.

### 💻 코드 구현
```python
def calculate_expected_uplift(cate, k_percent):
    cutoff = int(n * k_percent / 100)
    
    # 1. 효과가 좋은 상위 K% 고객의 CATE 점수를 합산
    model_uplift = np.sum(cate_sorted[:cutoff])
    
    # 2. 랜덤 타겟팅의 기대값 계산
    random_uplift = (k_percent / 100.0) * np.sum(cate)
    
    return int(round(model_uplift)), int(round(random_uplift))
```

### 🔍 원리 설명
* **상위 고객 합산 (Targeting Uplift):** CATE 점수는 "이 고객에게 광고 메시지를 보냈을 때 구매 확률이 몇 %나 상승하는가"를 나타냅니다. 따라서 정렬 후 상위 $K$\% 고객의 점수를 모두 합산(`np.sum`)하면 **해당 집단 전체에게 집중 마케팅을 집행했을 때 실제로 늘어날 총 추가 구매수**를 직접 예측할 수 있습니다.
* **랜덤 기대값 (Random Baseline):** 아무 기준 없이 무작위로 $K$\% 고객을 선정하여 메시지를 보내면, 전체 타겟팅 시의 총 기대효과(`np.sum(cate)`)에 $K$\% 비율을 곱한 만큼만 정비례하여 판매량이 증가합니다.

---

## 4. `calculate_negative_cate_avoidance` (역효과 방어율 계산)
> **역할:** 메시지 폭탄이나 불필요한 광고로 인해 오히려 반발심을 가지고 이탈(CATE < 0)할 위험 고객을 모델이 얼마나 잘 격리하여 방어했는지를 측정합니다.

### 💻 코드 구현
```python
def calculate_negative_cate_avoidance(cate, threshold_pct=15):
    # 1. 모델이 예측한 음수(역효과) 고객 정의
    is_neg = (cate < 0)
    total_neg = np.sum(is_neg)
    
    # 2. 효과가 가장 나쁜 하위 15% 추출
    order_asc = np.argsort(cate)
    cutoff = int(n * threshold_pct / 100)
    bottom_indices = order_asc[:cutoff]
    
    # 3. 하위 15% 영역 내에 실제 포착된 음수 고객 수 계산
    neg_avoided = np.sum(is_neg[bottom_indices])
    raw_rate = neg_avoided / total_neg
    
    # 4. 무작위 차단선(P)을 빼주어 최종 정규화
    p = threshold_pct / 100.0
    norm_rate = (raw_rate - p) / (1.0 - p)
    
    return int(neg_avoided), raw_rate, max(0.0, norm_rate)
```

### 🔍 원리 설명
* **음수 정의 (Targeting Sleeping Dogs):** 광고를 안 받는 게 나은 역효과군(`cate < 0`)을 식별하고 총 인원을 구합니다.
* **하위 격리 (Isolation):** CATE가 낮을수록(오름차순) 정렬하여 최악의 효과가 우려되는 하위 $P$\% (기본값 15%) 영역을 추출합니다.
* **방어율 정규화 (Min-Max Scaling):** 무작위로 15%를 제외해도 통계적으로 15%의 음수 유저는 우연히 필터링됩니다. 따라서 무작위 성과선($p=0.15$)을 차감한 뒤 정규화 공식을 적용하여 **"아무 조건 없이 랜덤하게 대상을 뺐을 때(0%) 대비 모델이 얼마나 더 능동적이고 정교하게 역효과 고객들을 하위 영역에 모아두어 차단에 성공했는가"**를 계산합니다.
  $$\text{Normalized Rate} = \frac{\text{Raw Recall Rate} - P}{1.0 - P}$$
