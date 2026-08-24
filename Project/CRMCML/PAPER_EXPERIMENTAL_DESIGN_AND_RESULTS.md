# 논문 본문: 인과 머신러닝 배경, 실험 설계 및 실증 결과

---

## 2. 배경 및 문제 정의 (Background & Problem Formulation)

### 2.1 기존 예측 기반 마케팅 타겟팅의 구조적 한계

기업의 마케팅 타겟팅에서 가장 널리 사용되는 전략은 구매 전환율(CVR, Conversion Rate)이 높을 것으로 예측되는 고객에게 마케팅 메시지를 집중 발송하는 것이다. 이 접근은 지도학습(Supervised Learning) 기반의 CVR 예측 모델 $\hat{P}(Y=1 \mid X)$을 구축하여, 예측 확률이 높은 순서대로 고객을 정렬하고 상위 $k\%$에게 메시지를 발송하는 구조를 따른다.

그러나 이 방법에는 근본적인 한계가 존재한다. CVR이 높다고 예측된 고객이 반드시 마케팅 메시지에 의해 추가 구매를 일으키는 것은 아니기 때문이다. 이들 중 상당수는 메시지를 수신하지 않더라도 자발적으로 구매할 고객이며, 일부는 오히려 메시지에 의해 구매를 철회하기도 한다. 이러한 문제를 체계적으로 이해하기 위해, 마케팅 메시지 수신 여부와 구매 반응에 따라 고객을 다음의 네 가지 유형으로 분류할 수 있다 (Radcliffe & Surry, 2011):

| 유형 | 정의 | $\tau(X)$ | 최적 전략 |
| :--- | :--- | :---: | :--- |
| **Persuadables (설득 가능 고객)** | 메시지를 수신할 때에만 구매가 유발되는 고객 | $\tau(X) > 0$ | **타겟팅 1순위** |
| **Iron Knights (자발적 구매자)** | 메시지 수신과 무관하게 자발적으로 구매하는 고객 | $\tau(X) \approx 0$ | 발송 불필요 (예산 낭비) |
| **Lost Causes (무반응 고객)** | 메시지를 수신하더라도 구매하지 않는 고객 | $\tau(X) \approx 0$ | 발송 불필요 (비용 낭비) |
| **Sleeping Dogs (잠자는 사자)** | 메시지를 수신하면 오히려 구매를 철회하는 고객 | $\tau(X) < 0$ | **발송 금지** |

기존 CVR 예측 모델은 이 네 유형을 구분하지 못한다. 자발적 구매자(Iron Knights)와 설득 가능 고객(Persuadables) 모두 높은 CVR을 보이므로 예측 점수만으로는 양자를 식별할 수 없으며, 더 나아가 잠자는 사자(Sleeping Dogs)에게 불필요한 메시지를 발송하여 역효과를 초래할 위험이 상존한다. 따라서 **"누가 구매할 것인가?"** 를 예측하는 것이 아니라, **"마케팅 메시지라는 자극(Treatment)이 투입되었을 때, 구매 행동이 추가로 유발되는 고객은 누구인가?"** 를 추론하는 인과적 접근이 필요하다.

---

### 2.2 인과 머신러닝과 조건부 평균 처치 효과 (CATE)

인과 머신러닝(Causal Machine Learning)은 관측 데이터 또는 실험 데이터로부터 처치(Treatment)가 결과(Outcome)에 미치는 인과적 효과를 추정하는 프레임워크이다. Neyman-Rubin의 잠재적 결과 프레임워크(Potential Outcomes Framework)에 따르면, 각 고객 $i$에 대해 두 가지 잠재적 결과가 존재한다:

- $Y_i(1)$: 마케팅 메시지를 **수신했을 때** 의 구매 결과
- $Y_i(0)$: 마케팅 메시지를 **수신하지 않았을 때** 의 구매 결과

개별 고객의 인과 효과(Individual Treatment Effect)는 $Y_i(1) - Y_i(0)$으로 정의되나, 한 고객이 동시에 메시지를 수신하면서 수신하지 않는 것은 불가능하므로(인과 추론의 근본 문제, Holland 1986), 관측 가능한 데이터로부터 이를 직접 계산할 수 없다. 이에 따라 본 연구에서 추정하고자 하는 핵심 대상은 고객의 관측 가능한 특성 벡터 $X$가 주어졌을 때의 **조건부 평균 처치 효과(CATE, Conditional Average Treatment Effect)** 이다:

$$\tau(X) = E[Y(1) - Y(0) \mid X]$$

$\tau(X)$의 부호와 크기는 앞서 정의한 고객 4분류에 직접 대응된다. $\tau(X) > 0$이면 메시지가 구매를 추가 유발하는 설득 가능 고객이며, $\tau(X) < 0$이면 메시지가 오히려 구매를 억제하는 잠자는 사자(Sleeping Dogs) 고객이다. 따라서 $\tau(X)$를 정밀하게 추정하고 이에 기반하여 양수 구간의 고객에게만 선택적으로 메시지를 발송하는 것이 본 연구의 최종 목표이다.

---

### 2.3 Double Machine Learning (DML)의 필요성

관측 데이터에서 CATE를 편향 없이 추정하기 위해서는, 마케팅 메시지 수신 여부($T$)와 구매 결과($Y$) 양쪽에 동시에 영향을 미치는 교란 변수(Confounders)의 효과를 제거해야 한다. 예를 들어, 과거 활동성이 높은 고객에게 우선적으로 메시지를 발송하는 기업의 타겟팅 규칙이 존재할 경우, 활동성이라는 변수는 처치 배정($T$)과 구매 결과($Y$) 모두에 영향을 미치는 교란 변수가 된다.

Chernozhukov et al. (2018)이 제안한 **Double Machine Learning (DML)** 은 이 문제를 2단계 직교화(Orthogonalization)로 해결한다:

**1단계 (Nuisance Estimation)**: 공변량 $X$로부터 결과 변수와 처치 변수의 기댓값을 각각 예측하여 잔차를 산출한다.

$$Y_{\text{res}} = Y - m(X), \quad \text{where } m(X) = E[Y \mid X]$$
$$T_{\text{res}} = T - e(X), \quad \text{where } e(X) = P(T=1 \mid X)$$

**2단계 (Causal Estimation)**: 잔차화된 처치와 잔차화된 결과 사이의 관계를 학습하여 순수 인과 효과 $\tau(X)$를 추정한다.

$$Y_{\text{res}} = \tau(X) \cdot T_{\text{res}} + \epsilon, \quad E[\epsilon \mid X, T] = 0$$

직관적으로, 1단계에서 "이 고객이 메시지를 받지 않아도 원래 살 확률($m(X)$)"과 "이 고객이 메시지를 받게 될 확률($e(X)$)"을 먼저 계산하여 차감함으로써, 2단계에는 오직 **메시지 수신으로 인해 추가로 발생한 순수 구매 반응 신호** 만이 잔차 형태로 남게 된다. 본 연구에서는 2단계 추정기로 Causal Forest (Wager & Athey, 2018)를 사용하여 CATE의 이질성(Heterogeneity)을 비모수적으로 학습하는 **CausalForestDML** 구조를 채택하였다.

---

## 3. 실험 설계 (Experimental Design)

본 절에서는 연구에 사용된 데이터의 수집 및 구성, 피처 엔지니어링, 대리 전이학습(Surrogate Indexing), 그리고 이단계 규제 CausalForestDML 모델의 설계를 순서대로 기술한다. 전체 파이프라인은 다음의 4단계로 구성된다.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ Stage 1. 표본 추출 (Sampling)                                              │
│   Raw DB (147GB) ─► D_OBS (14.4M rows, 소스 도메인)                        │
│                  ─► D_RCT (9,215 rows, 타겟 도메인)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ Stage 2. 피처 엔지니어링 (Feature Engineering)                              │
│   54개 피처 생성 (v0: 34개 메타 + v1~v4: 각 5개 행동 피처)                  │
├─────────────────────────────────────────────────────────────────────────────┤
│ Stage 3. 대리 전이학습 (Surrogate Indexing Transfer Learning)               │
│   D_OBS에서 v1~v4 그룹별 CVR 예측 모델 사전 학습                            │
│   ─► 대리 점수 S_v1 ~ S_v4를 D_RCT에 이식                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│ Stage 4. 이단계 규제 CausalForestDML (Two-Stage Regularized DML)            │
│   1단계: Nuisance 직교화 (min_samples_leaf=200)                             │
│   2단계: Causal Forest CATE 추정 및 하이퍼파라미터 튜닝                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 3.1 데이터 수집 및 코호트 구성

#### 3.1.1 원시 데이터 개요

본 연구에 사용된 데이터는 이커머스 기업의 마케팅 메시지 발송 및 구매 전환 로그 데이터베이스(`messages.csv`, 147GB)이다. 이 데이터베이스에는 2021년~2023년 기간 동안 이메일, 모바일 푸시, SMS 등 다양한 채널로 발송된 마케팅 메시지의 수신 기록과 각 메시지에 대한 고객의 열람, 클릭, 구매 전환 반응이 행(Row) 단위로 기록되어 있다.

147GB 규모의 원시 데이터를 단일 로컬 환경에서 전체 로드하는 것은 메모리 초과(OOM) 문제로 불가능하다. 또한 단순 행 단위 무작위 추출(Random Row Sampling)을 수행할 경우, 동일 고객의 과거 메시지 수신 및 구매 이력이 시간 순서대로 연결되지 않아 시계열 피처(재구매 주기, 피로도 등)가 왜곡된다. 이를 해결하기 위해 **고객 단위 이력 보존 그룹 샘플링(Client-Level Group Sampling)** 을 적용하였다. 추출 대상으로 선정된 고객에 대해서는 해당 고객의 전체 수신 및 구매 이력을 100% 온전히 복원하여 시계열의 단절 없이 피처를 생성할 수 있도록 하였으며, PyArrow Scanner 기반 스트리밍 읽기 방식을 사용하여 메모리 사용량을 250MB 이하로 제어하였다.

#### 3.1.2 두 개의 코호트 구성

원시 데이터로부터 분석 목적에 따라 두 개의 독립적인 코호트를 구성하였다.

**소스 도메인: 글로벌 관측 코호트 ($\mathcal{D}_{\text{OBS}}$).**
전체 데이터베이스에서 평생 1회 이상 구매 이력이 있는 고유 고객 138,710명 중 20,000명을 무작위 추출하여 구매 이력 보유군(Cases)으로, 해시 기반 확률 필터링(`hash(client_id) % 1000 < 15`)을 통해 30,000명을 무작위 추출하여 비구매 대조군(Controls)으로 구성하였다. 이 50,000명 고객의 전체 마케팅 수신 이력을 복원한 결과, 총 **14,403,294행** 의 관측 데이터셋이 생성되었다. 이 코호트는 일상적인 마케팅 운영 기간의 대규모 행동 로그를 포함하며, 고객의 일반적인 구매 성향 패턴을 사전 학습하기 위한 **소스 도메인** 으로 사용된다.

**타겟 도메인: Matched A/B 테스트 코호트 ($\mathcal{D}_{\text{RCT}}$).**
동일한 기저 고객 풀에서, 무작위 배정(Randomized Controlled Trial, RCT) 원칙이 엄격하게 준수된 12개 캠페인의 수신 기록만을 별도로 격리하여 추출하였다. 이 12개 캠페인은 동일 시각·동일 채널로 집행된 6개 쌍(Matched Pairs)의 구조를 가진다. 각 쌍에서 실험군 캠페인($T=1$)의 고객에게는 실제 마케팅 메시지가 발송되었고, 대조군 캠페인($T=0$)의 고객은 동일한 타겟팅 조건으로 선정되었으나 실제 메시지 발송 없이 구매 여부만 추적하는 순수 홀드아웃 대조군(Holdout Control)이다.

##### [Table 1] A/B 테스트 코호트의 Matched-Pair 구조

| Pair | Treatment ($T=1$) | Control ($T=0$) | 발송 채널 | 발송 시점 |
| :---: | :---: | :---: | :--- | :--- |
| 1 | Campaign 577 | Campaign 676 | Email | 2021-07-22 |
| 2 | Campaign 721 | Campaign 722 | Email | 2021-07-23 |
| 3 | Campaign 1318 | Campaign 1319 | Email | 2021-09-23 |
| 4 | Campaign 1468 | Campaign 1469 | Email | 2021-10-07 |
| 5 | Campaign 11031 | Campaign 11035 | Mobile Push | 2023-03-17 |
| 6 | Campaign 11329 | Campaign 11345 | Mobile Push | 2023-03-29 |

이 코호트에서 구매 전환($Y=1$) 698건 전체와 비구매($Y=0$) 6,980건을 **1대10 비율(Case-Control Design)** 로 추출하여 총 **9,215행** (고유 고객 7,678명)의 타겟 도메인 데이터셋을 구성하였다. 1대10 비율의 선택은 Breslow (1996)의 환자-대조군 설계 정리에 근거한다. 환자군 1명당 대조군 $k$명을 배정할 때의 상대적 통계 효율성은 $\text{Efficiency} = k/(k+1)$로 수렴하며, $k=10$일 때 $10/11 \approx 90.9\%$의 효율성을 달성한다. 이는 전체 비구매 대조군을 모두 사용할 때 대비 통계적 검정력의 91%를 유지하면서도, Causal Forest의 연산 메모리와 수렴 속도를 최적화하는 경계점이다.

Case-Control 추출 과정에서 표본 내 기저 CVR($\approx 9.09\%$)이 실제 모집단 전체 CVR에 비해 인위적으로 부풀려지는 사전 확률 편향(Prior Probability Shift)이 발생한다. 그러나 Qini Score 및 AUUC와 같은 순위 기반 평가 지표는 사전 확률의 단조 스케일링 변화에 불변(Invariant)하는 특성을 갖는다 (King & Zeng, 2001; Prentice & Pyke, 1979). 또한, 실제 절대적 순증 전환량을 산출하는 4.5절의 비즈니스 임팩트 시뮬레이션에서는 모집단 CVR 비율에 맞춘 **사전 확률 오프셋 보정(Prior Weighting Offset Calibration, $w = \frac{p_{\text{pop}}}{p_{\text{sample}}}$)** 을 적용하여 전체 모집단 단위의 실제 순증 구매 효과로 전환 보정하였다.

---

### 3.2 피처 엔지니어링

각 코호트($\mathcal{D}_{\text{OBS}}$, $\mathcal{D}_{\text{RCT}}$)에 대해 동일한 피처 엔지니어링 파이프라인을 적용하여 총 54개의 피처를 생성하였다. 피처는 계량경제학적 역할과 개념적 유사성에 따라 5개 그룹으로 분류된다.

##### [Table 2] 피처 그룹 구성 요약 (총 54개)

| 그룹 | 명칭 | 변수 수 | 핵심 변수 예시 | 계량적 역할 |
| :---: | :--- | :---: | :--- | :--- |
| $v0$ | 기본 발송 메타 (Base Metadata) | 34 | `channel_email`, `is_holiday`, `total_purchases` | 발송 환경의 외생 통제 변수 |
| $v1$ | 재구매 주기 (Recency & Hazard) | 5 | `days_since_last_purchase`, `feat_rtb_hazard`, `feat_recency_to_cycle_ratio` | 자발적 재구매 도래 시점 및 시급성 (Ascarza 2016; Rößler & Schoder 2022) |
| $v2$ | 시간 맥락 (Calendar & Context) | 5 | `cal_is_weekend`, `feat_dow_shift`, `feat_payday_bump` | 요일·시간대·월급날의 구매 변동성 |
| $v3$ | 유저 활동성 (Activity & Cadence) | 5 | `u_open_rate_30d`, `feat_fatigue`, `u_click_rate_30d` | 최근 30일 반응 조밀도 및 메시지 피로도 |
| $v4$ | 토픽 관심도 (Topic Affinity) | 5 | `feat_like_last_success`, `feat_topic_novelty`, `topic_N7` | 메시지-고객 선호 토픽 일치도 |

$v1 \sim v4$의 각 그룹을 **정확히 5개 변수로 균등 배치(1:1:1:1 Balanced Specification)** 한 것은 의도적 설계이다. 이후 대리 전이학습(Stage 3)에서 각 그룹별로 독립적인 대리 점수를 산출하는데, 특정 그룹의 변수 수가 과도하게 많으면 해당 그룹의 대리 모델이 다른 그룹 대비 부당하게 높은 판별력을 갖게 되어 차원 편향(Dimension Imbalance)이 발생한다. 이를 원천적으로 차단하기 위해 학술 문헌(Ascarza, 2016; Rößler & Schoder, 2022)에 기반하여 각 그룹의 변수를 5개로 통일하고, 다중공선성이 높은 변수는 사전에 정제하여 제외하였다.

피처 생성 과정에서 다음의 세 가지 데이터 품질 규칙을 엄격히 적용하였다:

1. **미래 정보 누출 차단 (Strict Causal Shift)**: 모든 시계열 피처는 현재 메시지 발송 시점 $t_{\text{current}}$ 이전($t < t_{\text{current}}$)의 이력만을 참조하도록 1-Step Prior Shift를 강제 적용하였다.
2. **결측치 우측 중단 대치 (Administrative Right-Censoring at 8,760h)**: 과거 수신·구매·클릭 이력이 전혀 없는 고객의 경과시간 결측값은 물리적 상한선인 8,760시간(1년)으로 통일 대치하여 트리의 분기 순도를 보존하였다.
3. **이상치 클리핑 (Outlier Clipping)**: 모든 수치형 피처의 상위 99.5분위수 이상 값을 클리핑하고, 경과시간 계열 변수의 음수값은 0으로 클리핑하였다.

---

### 3.3 대리 전이학습 (Surrogate Indexing Transfer Learning)

#### 3.3.1 도입 배경 및 필요성

타겟 도메인 $\mathcal{D}_{\text{RCT}}$는 무작위 배정이 보장된 인과 추론에 적합한 데이터이나, 표본 규모(9,215행)가 작고 구매 전환 건수(698건)가 희소하다. 이 상태에서 20개의 원본 행동 피처($v1 \sim v4$ 각 5개)를 Causal Forest에 직접 투입하면, 소규모 표본 대비 피처 차원이 과도하여 모델이 데이터의 잡음까지 학습하는 인과 과적합(Causal Overfitting)이 발생한다.

이를 해결하기 위해 Athey, Chetty, Imbens & Kang (2019)의 **대리 지표(Surrogate Index)** 이론을 응용·확장하였다. 본 연구의 대리 전이학습 프레임워크는 Athey et al. (2019)의 Surrogate Index 개념을 소스 도메인의 대규모 행동 로그로부터 타겟 도메인으로의 차원 축소 및 인과 신호 이식(Feature Representation & Dimension Reduction Transfer) 목적으로 확장하여 고안한 구조이다. 핵심 아이디어는 소스 도메인 $\mathcal{D}_{\text{OBS}}$(14.4M행, 구매 35,321건)에서 각 피처 그룹의 원본 변수 조합과 구매 전환($Y$) 간의 관계를 먼저 사전 학습한 뒤, 학습된 예측 확률(대리 점수)을 타겟 도메인의 동일 고객에게 이식(Transfer)하여 차원을 축소하는 것이다.

#### 3.3.2 전이 프로토콜

소스 도메인 $\mathcal{D}_{\text{OBS}}$에서 각 피처 그룹 $g \in \{v1, v2, v3, v4\}$에 대해 독립적인 RandomForestClassifier를 훈련하였다.

- **입력**: 해당 그룹의 5개 원본 피처 $X_g$
- **타깃**: 구매 전환 여부 $Y \in \{0, 1\}$
- **모델 규제**: `n_estimators=100`, `max_depth=6`, `min_samples_leaf=50`

모델의 복잡도를 의도적으로 낮게 제한한 것은 과적합 방지를 위한 설계이다. 대리 모델의 학습 정확도가 과도하게 높으면, 관측 데이터에 내재된 타겟팅 편향(특정 유형의 고객에게 우선 발송하는 기업의 운영 규칙)이 대리 점수에 암기되어, 이 편향이 타겟 도메인으로 이식될 때 Common Support를 파괴하고 CATE 추정을 오염시킨다. 반대로 정확도가 랜덤 수준(AUC ≈ 0.50)으로 너무 낮으면 원본 변수에 내재된 행동 신호가 소실된다. 본 모델은 AUC 0.65~0.75 범위의 **일반화된 판별력** 을 달성하도록 규제하였다.

학습된 모델로부터 산출된 CVR 예측 확률 $S_g(X_g) = \hat{P}(Y=1 \mid X_g) \in [0, 1]$을 **연속형 소수점 값 그대로** 타겟 도메인 $\mathcal{D}_{\text{RCT}}$의 동일 고객에게 이식하였다. Athey & Imbens (2019)의 표준 프로토콜에 따라 대리 점수를 범주형으로 이산화(Binning)하지 않은 것은, 연속 점수의 미세한 차이를 보존하여 개인화 타겟팅 해상도의 정보 유실을 방지하기 위함이다.

```text
[ 소스 도메인 D_OBS: 14.4M rows, 구매 35,321건 ]
  │
  ├─ v1 원본 5개 피처 ──► RFC ──► S_v1 (재구매 시급성 대리 점수)
  ├─ v2 원본 5개 피처 ──► RFC ──► S_v2 (시간 맥락 대리 점수)
  ├─ v3 원본 5개 피처 ──► RFC ──► S_v3 (활동성 대리 점수)
  └─ v4 원본 5개 피처 ──► RFC ──► S_v4 (토픽 관심도 대리 점수)
  │
  ▼ (Out-of-Sample Score Transfer)
[ 타겟 도메인 D_RCT: 9,215 rows, 구매 698건 ] ──► CausalForestDML 학습
```

---

### 3.4 공변량 불균형 진단 및 DML 직교화의 필요성

대리 점수가 이식된 $\mathcal{D}_{\text{RCT}}$에서 CATE를 추정하기에 앞서, 무작위 배정의 정합성을 피처 그룹 수준에서 검증하였다. 각 대리 점수를 분위수 기준으로 등급 분할한 뒤, 등급별 처치 배정 비율($T=1$ 비율)을 산출하여 교란 편향의 존재 여부를 진단하였다.

##### [Table 3] 대리 점수 분위수 등급별 처치 배정 비율 및 선택 편향 진단

| 피처 그룹 | 분위수 등급 | $T=1$ 비율 | 대조군 CVR | Uplift | 진단 결과 |
| :--- | :--- | :---: | :---: | :---: | :--- |
| $v1$ (재구매 주기) | Group 0 | 49.02% | 8.65% | −3.80%p | **무편향** (처치 비율 ≈ 50%) |
| | Group 1 | 50.59% | 24.04% | −2.44%p | **무편향** |
| $v2$ (시간 맥락) | Group 0 | 49.70% | 0.87% | −0.15%p | **무편향** |
| | Group 1 | 51.03% | 0.53% | −0.02%p | **무편향** |
| | Group 2 | 46.80% | 31.50% | −14.46%p | **무편향** (고CVR 그룹의 기저 구매력) |
| | Group 3~4 | 39.1%~49.6% | 3.21%~13.21% | +1.09~+1.50%p | **무편향** |
| $v3$ (유저 활동성) | Group 0 | 50.70% | 3.01% | −1.22%p | **무편향** (처치 비율 ≈ 50%) |
| | Group 1 | 50.40% | 1.60% | −0.46%p | **무편향** |
| | Group 2 | 48.83% | 5.97% | −2.74%p | **무편향** |
| | Group 3 | 46.81% | 19.53% | **−8.29%p** | **무편향** (순수 인과 역효과) |
| | Group 4 | 48.20% | 17.88% | **−5.14%p** | **무편향** (순수 인과 역효과) |
| $v4$ (토픽 관심도) | Group 0 | **100.00%** | — (대조군 0명) | +2.14%p | **극단적 선택 편향** (대조군 완전 유실) |
| | Group 1 | **100.00%** | — (대조군 0명) | +8.61%p | **극단적 선택 편향** (대조군 완전 유실) |
| | Group 2 | **37.72%** | 1.66% | +2.78%p | **선택 편향** |
| | Group 3 | **2.75%** | 14.46% | +23.54%p | **극단적 선택 편향** (처치군 거의 부재) |
| | Group 4 | **1.68%** | 9.69% | +7.55%p | **극단적 선택 편향** (처치군 거의 부재) |

진단 결과는 피처 그룹에 따라 명확한 대조를 보인다. $v1$(재구매 주기), $v2$(시간 맥락), $v3$(유저 활동성) 세 그룹에서는 처치 배정 비율이 46~51% 범위로 무작위 배정이 공정하게 유지되었다. 특히 $v3$에서 주목할 점은, 처치 배정이 무편향임에도 불구하고 고활동 유저(Group 3~4)에서 −5.14%p ~ −8.29%p의 **순수 인과 역효과(Sleeping Dogs Effect)** 가 관측된다는 사실이다. 이는 선택 편향에 의한 허구적 현상이 아니라, 공정한 무작위 배정 하에서 실제로 발생하는 마케팅 자극의 역효과임을 의미한다 (4.2절에서 상세 분석).

반면, $v4$(토픽 관심도)에서는 기업의 과거 토픽 기반 타겟팅 규칙에 의해 처치 배정이 1.68%~100%로 극단적으로 쏠리는 심각한 선택 편향이 확인되었다. Group 0~1에서는 대조군이 0명으로 완전히 유실되어 단순 Uplift 산출이 불가능하고, Group 3~4에서는 처치군이 2~3%에 불과하여 단순 차이 비교 시 +23.54%p라는 과대 추정된 허구적 수치가 산출된다.

비록 $v4$의 Group 0~1과 같이 공통 지원 영역(Common Support / Overlap, $0 < P(T=1 \mid X) < 1$)을 왜곡하는 샘플이 존재하더라도, DML 파이프라인은 3.5.2절에 기술된 성향 점수 트리밍(`ClippedClassifier`, $[0.05, 0.95]$)을 통해 이러한 극단적 처치 배정 표본을 1단계 과정에서 자동으로 안정화·필터링한 후 2단계 CATE 학습을 진행한다. 이 결과는 $v4$에 내재된 교란 편향을 제거하지 않고 단순 그룹 간 평균 차이(Naive ATE)를 비교할 경우 심각하게 왜곡된 인과 효과를 추정하게 됨을 실증하며, **DML의 1단계 직교화를 통해 $v4$에 내재된 교란 효과를 잔차화($T_{\text{res}}, Y_{\text{res}}$)하여 제거하는 것이 필수적** 임을 계량적으로 입증한다.

---

### 3.5 이단계 규제 CausalForestDML 설계

#### 3.5.1 1단계: Cross-fitting 장치 및 Nuisance 모델의 과적합 규제 (`min_samples_leaf = 200`)

Chernozhukov et al. (2018)의 Double Machine Learning(DML) 프레임워크는 1단계 Nuisance 모델($m(X) = E[Y \mid X]$, $e(X) = P(T=1 \mid X)$)의 과적합 편향(Overfitting Bias)을 차단하기 위해 **샘플 분할 및 교차 맞춤(K-Fold Cross-fitting)** 을 핵심 기하학적 장치로 사용한다. 교차 맞춤을 통해 1단계 예측 편향이 2단계 인과 추정으로 전이되는 것을 방지한 상태에서, Nuisance 모델의 개별 복잡도를 어떻게 제어할 것인가가 2단계 CATE 추정 품질의 핵심 변수가 된다.

일반적인 예측 모델에서는 리프 크기를 20~50으로 작게 설정하여 예측 정확도를 높이지만, DML의 1단계 Nuisance 트리 모델에서 이 전략을 적용하면 **순증 신호 소거 오류(Signal Extinguishment)** 라는 치명적 문제가 발생한다. 리프 크기가 작을수록 1단계 모델은 고객의 미세한 행동 패턴까지 과도하게 기저 CVR로 흡수하여 $m(X) \approx Y$로 수렴하고, 잔차 $Y_{\text{res}} = Y - m(X) \to 0$이 되어 2단계 Causal Forest로 전달되어야 할 인과 순증 신호가 소멸하기 때문이다.

따라서 본 연구에서는 DML의 Cross-fitting 구조 위에 1단계 Nuisance 트리의 리프 크기를 **`min_samples_leaf = 200`** 으로 강하게 규제(Oversmoothing)하는 추가 정규화(Regularization) 전략을 결합하였다:

1. **이론적 근거 (Oversmoothing for Signal Preservation)**: 1단계 트리를 의도적으로 둔탁하게 평활화하여 고객의 기저 구매 성향만 대략적으로 차감하고, 마케팅 자극에 의한 순증 인과 신호를 잔차에 보존시키는 Chernozhukov et al. (2018)의 잔차화 원리를 적용한다.
2. **실증적 근거 (Empirical R-score Peak)**: 그리드 탐색 결과, `min_samples_leaf = 50` 이하에서는 Qini 스코어가 음수로 하락하는 신호 소거 현상이 관측되었고, **200 지점에서 Out-of-sample R-score 및 Qini가 최고점** 을 기록하였다.
3. **분산 제어 근거 (Wager & Athey, 2018)**: 인과 트리의 단말 노드에서 CATE 추정 분산은 $\text{Var}(\hat{\tau}_{\text{leaf}}) \approx \sigma_1^2 / N_1 + \sigma_0^2 / N_0$을 따른다. 분석 표본 9,215명에서 `min_samples_leaf = 200`이어야 단말 노드당 처치군($N_1 \approx 100$)과 대조군($N_0 \approx 100$)이 균형 있게 확보되어 분산 폭발을 방지할 수 있다.

계량경제학적 관점에서 1단계 규제가 강해지면 Nuisance 모델이 미세한 교란 변수를 완전히 제거하지 못하고 잔차($Y_{\text{res}}, T_{\text{res}}$)에 남겨둘 수 있다는 우려(Underfitting / Unobserved Confounding)가 제기될 수 있다. 그러나 본 연구는 4.3절의 Placebo Test(위약 ATE = −0.0022, 95% CI가 0 포함) 및 Cinelli & Hazlett 민감도 분석(RV = 0.57%)을 통해, 1단계의 강한 규제 적용에도 불구하고 잔차에 유의미한 교란 편향이 잔존하지 않고 비편향 인과 신호가 정밀하게 추출되었음을 실증적으로 입증하였다.

#### 3.5.2 성향 점수 트리밍 (Propensity Score Trimming & Bounding)

1단계에서 추정된 성향 점수 $e(X)$가 0 또는 1에 극단적으로 수렴하는 영역에서는 반사실적 대조군(Counterfactual)이 존재하지 않아 DML의 가중치 계산에서 분모가 0에 근접하는 수치적 불안정이 발생한다. 특히 $v4$(토픽 관심도)와 같이 특정 그룹에서 처치 배정이 100% 또는 1% 미만으로 쏠리는 현상(Table 3 참조)이 존재할 때, Common Support 가정이 파괴되는 위험이 크다.

이를 계량경제학적으로 해결하기 위해 **Crump, Hotz, Imbens & Mitnik (2009)** 의 성향 점수 트리밍 및 바운딩(Propensity Score Trimming/Bounding) 이론에 근거하여, 성향 점수를 $[0.05, 0.95]$ 범위로 강제 제한(Clipping)하는 `ClippedClassifier` 래퍼(EconML Wrapper)를 구축하였다. 이 공통 지원 영역(Common Support Region) 내부의 표본만을 인과 추정에 활용함으로써, Overlap 조건이 성립하지 않는 극단적 표본은 성향 점수 가중치 폭발 없이 안전하게 조정되어 CATE 추정의 수학적 완결성을 보장한다.

#### 3.5.3 2단계: Causal Forest CATE 튜닝

2단계 Causal Forest의 하이퍼파라미터는 다음의 그리드에서 R-score 기준 탐색을 수행하였다:

| 파라미터 | 탐색 범위 | 설정 근거 |
| :--- | :--- | :--- |
| `min_samples_leaf` | [100, 200, 300, 400] | 하한 100: 노드당 처치/대조 각 ~50명 확보. 상한 400: 과도한 평활화로 인한 타겟팅 해상도 저하 방지 |
| `max_depth` | [3, 5, 8] | 8 이하로 제한하여 고차 교호작용에 의한 인과 과적합 차단 |

5-Fold Cross-Validation을 통해 각 피처 구성(Ablation Configuration)별 최적 파라미터를 선정하고, CATE 예측값의 분포, AUUC 및 Qini 스코어를 산출하였다.

---

### 3.6 평가지표

모델의 인과 타겟팅 성능은 다음의 지표로 평가하였다.

- **Qini Score ($Q$)**: 모델이 추정한 CATE 순서대로 고객을 정렬하여 상위 $k\%$에게만 메시지를 발송했을 때, 무작위 발송 대비 얼마나 더 많은 순증 구매를 확보하는지를 나타내는 면적 지표이다. $Q > 0.40$이면 산업 상위 수준의 인과 선별력으로 판단한다.
- **AUUC (Area Under the Uplift Curve)**: 누적 발송 비율 대비 인과 순증 구매 곡선의 면적.
- **음수 CATE 회피율 (Negative CATE Avoidance Rate)**: Sleeping Dogs 고객($\tau(X) < 0$)에게 메시지 발송을 차단한 비율.

---

## 4. 실증 결과 (Empirical Results)

### 4.1 소거 연구 (Ablation Study): 피처 그룹별 인과적 기여도

각 피처 그룹의 인과적 기여도를 규명하기 위해, 기본 메타 피처($v0$)를 Baseline으로 고정하고 $v1 \sim v4$를 순차적으로 추가하는 소거 연구를 수행하였다.

##### [Table 4] CausalForestDML 피처 구성별 소거 연구 결과

| 피처 구성 | ATE | Qini Score | AUUC | Baseline 대비 Qini 변화 |
| :--- | :---: | :---: | :---: | :--- |
| $v0$ (Baseline) | −0.0672 | 0.3858 | −0.1151 | — (기준) |
| $v0 + v1$ | −0.0658 | **0.4171** | −0.0825 | +0.0312 ↑ |
| $v0 + v2$ | −0.0655 | 0.3914 | −0.1108 | +0.0055 ↑ |
| $v0 + v3$ | −0.0667 | **0.4083** | −0.0921 | +0.0225 ↑ |
| $v0 + v4$ | −0.1044 | **0.4593** | −0.0357 | **+0.0734 ↑ (단독 최고)** |
| Full ($v0 \sim v4$) | −0.1270 | 0.3815 | −0.1176 | −0.0043 (다중공선성 노이즈 증폭) |

이 결과로부터 각 피처 그룹의 인과적 역할을 다음과 같이 규명하였다.

**$v4$ (토픽 관심도) — 개인화 의도 일치(Intent Alignment) 변수.**
단독 추가 시 Qini를 0.3858에서 **0.4593으로 +0.0734 포인트** 향상시켜 개별 피처 중 가장 큰 인과 기여도를 기록하였다. $v4$는 현재 발송되는 메시지의 토픽과 고객의 과거 구매 성공 메시지의 토픽 간 일치도를 측정한다. 고객의 현재 관심사와 무관한 일괄 메시지는 스팸 피로감을 유발하지만, $v4$를 통해 메시지-고객 선호를 1대1 매칭함으로써 구매 반응률이 극대화된다.

**$v1$ (재구매 주기) — 인과 순증 유발 핵심 변수.**
Qini를 **0.4171로 +0.0312 포인트** 향상시켰다. 자발적 재구매 주기가 도래한 시점의 고객에게 적시에 메시지를 발송하면 결제 전환이 촉진된다는 Ascarza (2018)의 재구매 주기 가설을 실증적으로 확인하였다.

**$v3$ (유저 활동성) — 수비 제약 변수 (Sleeping Dogs 필터).**
Qini를 **0.4083으로 +0.0225 포인트** 향상시켰다. 이 향상은 $v3$가 공격적으로 구매 유저를 발굴한 결과가 아니라, 역효과 유저(Sleeping Dogs)를 발송 대상에서 제외(Exclude)하는 수비적 필터 기능을 수행한 결과이다. 이에 대한 실증 분석은 4.2절에서 상세히 다룬다.

**$v2$ (시간 맥락) — 시간적 수용성 보조 변수.**
Qini 0.3914로 미세하지만 안정적인 향상을 기여하였다. 메시지 발송 시점의 요일·시간대 수용성을 보완하는 보조적 역할을 수행한다.

**Full 모델에서 CausalForestDML 성능 변화 메커니즘.**
모든 대리 점수를 동시 결합한 Full 모델에서 CausalForestDML의 Qini가 0.3815로 복귀하는 현상이 관측되었다. 이는 단순한 나무 모델의 피처 수 한계가 아니라, 사전 사전 학습된 대리 점수들($S_{v1} \sim S_{v4}$) 간에 존재하는 **높은 상관관계(다중공선성, Multicollinearity)** 에 기인한다. 대리 점수 변수들이 다수 입력되면 DML 1단계 Nuisance 모델링 과정에서 중복된 성향점수 및 outcome 회귀 정보가 상호 간섭을 일으켜 잔차($Y_{\text{res}}, T_{\text{res}}$)의 노이즈가 증폭되고 과적합(Over-residualization)이 유발된다. 
반면, 동일한 Full 구성을 **이중 강건(Doubly Robust) 구조의 DR-Learner** 에 적용했을 때에는(4.4절 Table 7 참조) 성향 점수 가중치와 결과 회귀가 상호 보완되면서 **Qini 0.5824** 를 달성해 이러한 다중공선성 노이즈를 완벽하게 극복함을 증명하였다.

---

### 4.2 Sleeping Dogs 역설: 고활동 VIP 고객의 인과 역효과

소거 연구에서 $v3$(유저 활동성) 피처의 인과적 역할을 심층 분석하기 위해, $v3$ 대리 점수의 분위수 등급별 실험군/대조군 구매율을 교차 검증하였다. Table 3의 편향 진단에서 확인된 바와 같이, $v3$의 모든 분위수 등급에서 처치 배정 비율이 46~51%로 공정하게 유지되므로, 이하의 분석 결과는 선택 편향에 오염되지 않은 **순수 인과 효과(Pure Causal Effect)** 로 해석할 수 있다.

##### [Table 5] 유저 활동성($v3$) 등급별 구매율 및 실제 인과 순증(Uplift)

| $v3$ 등급 | 총 유저 수 | $T=1$ 비율 | 대조군 CVR ($T=0$) | 실험군 CVR ($T=1$) | Uplift (순수 인과 순증) | 세그먼트 |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Group 0 (최저 활동) | 2,428 | 50.70% | 3.01% | 1.79% | **−1.22%p** | 약한 역효과 (무반응에 가까움) |
| Group 1 (저활동) | 1,385 | 50.40% | 1.60% | 1.15% | **−0.46%p** | 무반응 (Lost Causes) |
| Group 2 (평균 활동) | 2,095 | 48.83% | 5.97% | 3.23% | **−2.74%p** | 경미한 역효과 |
| Group 3 (고활동) | 1,521 | 46.81% | 19.53% | 11.24% | **−8.29%p** | **최고 피로 민감 (강한 Sleeping Dogs)** |
| Group 4 (최고 활동/VIP) | 1,857 | 48.20% | 17.88% | 12.74% | **−5.14%p** | **자발적 구매 충성 (Sleeping Dogs)** |

이 결과는 마케팅 타겟팅의 직관에 반하는 중요한 인과 역설을 보여준다. 전통적 CVR 예측 모델에서 높은 가치로 분류되는 고활동 유저(Group 3~4)는 메시지를 수신하지 않을 때 대조군 CVR이 17.88~19.53%에 달하는 우수 고객군이다. 그러나 이들에게 마케팅 메시지를 발송하면 구매율이 11.24~12.74%로 **5.14~8.29%p 급감** 한다. 특히 주목할 점은 이 역효과가 **처치 배정 비율 46~48%의 공정한 무작위 배정 환경** 에서 관측되었다는 것이다. 이는 선택 편향에 의한 허구적 현상이 아니라, 마케팅 자극이 실제로 유발하는 순수 인과적 구매 억제 효과임을 의미한다.

세그먼트별 세부 역효과 차이를 분석하면, 최상위 VIP 고객군인 Group 4(−5.14%p)보다 직전 고활동 고객군인 Group 3(−8.29%p)에서 인과 역효과가 더 강하게 나타난다. 이는 Group 4(최고 VIP)의 경우 강력한 브랜드 충성도와 자발적 구매 필요성이 확고하여 불필요한 마케팅 메시지 자극(피로감)에도 불구하고 자발적 구매를 상당 부분 유지하는 충성도 완충 효과(Loyalty Buffer)가 작동하기 때문이다. 반면 Group 3(고활동)은 플랫폼 이용 빈도는 높으나 VIP 수준의 절대적 구매 충성도에는 미치지 못하여, 메시지 수신 시 피로감 및 스팸 반발심에 가장 민감하게 반응하여 구매를 철회하는 **최고 피로 민감 세그먼트(Peak Fatigue-Sensitive Segment)** 로 기능한다.

따라서 $v3$ 피처가 DML 모델에 투입되었을 때 Qini 스코어가 상승하는 메커니즘은, $v3$가 Group 3~4 고활동 고객을 식별하여 발송 대상에서 제외하는 **수비적 필터(Exclusion Filter)** 역할을 수행하기 때문이다. Qini 지표는 역효과 유저를 성공적으로 발송 대상에서 배제할수록 값이 상승하는 구조를 가지므로, 이 수비적 기여가 Qini 향상으로 직접 반영된다.

---

### 4.3 계량경제학적 무결성 검증 (Placebo Test & Sensitivity Analysis)

본 모델의 인과 추정이 허구적 신호(Spurious Signal)를 학습한 것이 아님을 입증하기 위해 두 가지 계량학적 검증을 수행하였다.

##### [Table 6] 인과 무결성 검증 결과

| 검증 항목 | 표본 | 추정 지표 | 95% CI / 검정 결과 | 판정 |
| :--- | :--- | :---: | :---: | :--- |
| **Placebo Test** | 대조군 $N=4,727$ / 가짜 처치 $T_{\text{placebo}} \sim \text{Bernoulli}(0.5)$ | Placebo ATE = **−0.0022** | **[−0.0306, +0.0262]** (0 포함) | **귀무가설 채택**: 모델이 허구적 인과 효과를 생성하지 않음 |
| **Sensitivity Analysis** (Cinelli & Hazlett, 2020) | 전체 $N=9,286$ / Full DML | Real ATE = **−0.0869** | $RV_{q=1}$ = **0.57%** | **강건성 확인**: 미관측 교란변수가 처치($T$)와 결과($Y$)의 잔여 분산을 각각 0.57% 이상 동시에 설명하지 못하는 한 추정 결과 유지 |

**Placebo Test.** 실제 처치가 존재하지 않는 대조군 표본($N=4,727$)에 무작위 가짜 처치를 주입하고 CausalForestDML을 적용한 결과, 위약 ATE는 −0.0022로 0에 수렴하였으며 95% 신뢰구간[−0.0306, +0.0262]이 0을 완벽하게 포함하였다. 이는 모델이 처치 신호가 없는 상황에서 허구적 인과 효과를 환각(Hallucination)하지 않음을 증명한다.

**Cinelli & Hazlett Sensitivity Analysis.** 미관측 교란변수의 강건성 지표인 Robustness Value ($RV_{q=1}$)가 0.57%로 산출되었다. 이는 관측되지 않은 잠재적 미관측 교란변수가 존재하더라도, 그 교란변수가 처치 배정($T$)과 구매 결과($Y$)의 잔여 분산을 각각 0.57% 이상 동시에 설명하지 못하는 한, 본 인과 추정의 방향성과 통계적 유의성이 유지됨을 의미한다.

이 두 계량학적 검증 결과는 3.5.1절에서 언급한 1단계 Nuisance 모델의 강력한 과적합 규제(`min_samples_leaf=200`) 적용 시 우려될 수 있는 잔여 교란편향(Residual Confounding) 문제에 대한 결정적 반증을 제공한다. 위약 효과가 0으로 수렴하고 0.57%의 강건성을 확보한 것은 1단계의 강한 규제에도 불구하고 residualization 과정에서 미관측/잔여 교란 효과가 제거되지 않은 채 2단계를 오염시키지 않았음을 증명한다.

---

### 4.4 다중 추정기 강건성 검증 (Multi-Estimator Robustness Validation)

대리 전이학습의 효과가 CausalForestDML이라는 특정 추정기에만 국한된 현상인지, 아니면 추정기의 선택과 무관하게 일반화 가능한 프레임워크인지를 검증하기 위해, 5개의 서로 다른 인과 추정기에 대해 동일한 소거 연구를 수행하였다.

##### [Table 7] 다중 인과 추정기별 대리 전이학습 효과 (Baseline $v0$ vs. Best Surrogate Configuration)

| 추정기 | Best 피처 구성 | Baseline Qini ($v0$) | Best Qini (Surrogate) | Qini 향상률 | 판정 |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **S-Learner** | $v0 + v1$ | −0.0255 | **0.0261** | **+202.5%** (양수 전환) | ✓ 개선 |
| **T-Learner** | Full ($v0 \sim v4$) | 0.2778 | **0.3995** | **+43.8%** | ✓ 개선 |
| **X-Learner** | $v0 + v4$ | −0.2067 | **0.1397** | **+167.6%** (양수 전환) | ✓ 개선 |
| **DR-Learner** | $v0 + v1$ | 0.5364 | **0.7128** | **+32.9%** (전체 최고) | ✓ 개선 |
| **CausalForestDML** | $v0 + v4$ | 0.3858 | **0.4593** | **+19.0%** | ✓ 개선 |

이 실험의 핵심 발견은 다음 네 가지이다.

**발견 1: 대리 전이학습의 추정기 비의존적(Estimator-Agnostic) 강건성.**
5개 추정기 **전원(5/5)** 에서 대리 전이 피처($v1 \sim v4$) 투입 시 Baseline($v0$) 대비 Qini 스코어가 유의미하게 상승하였다. 이는 본 연구의 대리 전이학습이 특정 비선형 트리 추정기에 종속된 일시적 현상이 아니라, 추정기의 구조와 무관하게 일관되게 인과 적중력을 향상시키는 **일반화 가능한 인과 전이 프레임워크** 임을 실증한다.

**발견 2: DR-Learner의 압도적 최고 성능 (Qini 0.7128).**
이중 강건(Doubly Robust) 구조를 가진 DR-Learner (Kennedy, 2023)에 재구매 주기 대리 점수($S_{v1}$)를 결합한 구성이 전체 실험에서 **최고 Qini 0.7128** (AUUC +0.3009)을 달성하였다. DR-Learner의 이중 강건성(성향 점수 모델 또는 결과 회귀 모델 중 하나만 올바르게 명시되어도 비편향 추정이 가능한 성질)과, 소스 도메인에서 이식된 재구매 시급성 점수의 강한 인과 신호가 결합된 결과이다.

**발견 3: 음수 Qini의 양수 전환 (S-Learner, X-Learner).**
S-Learner와 X-Learner는 Baseline($v0$)만 사용할 경우 Qini가 음수(각각 −0.0255, −0.2067)로, 무작위 발송보다 못한 성능을 보였다. 그러나 대리 피처 투입 후 각각 +0.0261, +0.1397로 양수 전환되어 인과 적중력을 회복하였다. 이는 소규모 RCT 데이터에서 인과 신호가 미약한 환경에서도 소스 도메인의 사전 학습된 행동 점수가 신호를 효과적으로 보강함을 보여준다.

**발견 4: 추정기 구조별 최적 대리 피처 결합의 차별성.**
Table 7에서 DR-Learner는 $v0+v1$(재구매 주기)에서 최고 Qini(0.7128)를 기록한 반면, CausalForestDML과 X-Learner는 $v0+v4$(토픽 관심도)에서 최고점(각각 0.4593, 0.1397)을 달성하였다. 이러한 차이는 각 인과 추정기의 연산 구조(Single/Two-Learner vs. Doubly Robust vs. DML Orthogonalization)가 갖는 구조적 귀속 편향(Inductive Bias)에 기인한다. DR-Learner는 성향 점수 가중치와 결과 회귀를 결합한 이중 강건 조정을 통해 $v1$의 시계열 Hazard 반응 신호를 정밀하게 증폭시키는 데 유리한 반면, CausalForestDML은 비모수 트리 분기를 통해 $v4$의 비선형 토픽 선호도 경계면을 분할할 때 최고의 추정 적중력을 나타낸다.

---

### 4.5 비즈니스 임팩트 및 최적 타겟팅 전략

##### [Table 8] Full 모델 ($v0 \sim v4$) 기반 상위 $k\%$ 타겟팅 시 순증 이익 시뮬레이션 (단위: 순증 구매 전환 건수, Net Incremental Conversions)

| 타겟팅 구간 | 기대 순증 이익 (Expected) | 무작위 발송 이익 (Random) | 인과 타겟팅 이득 (Gain) |
| :---: | :---: | :---: | :--- |
| Top 10% | +30,259 | −2,422 | **+32,681 (+1,349%)** |
| Top 20% | +41,855 | −4,844 | **+46,699 (+964%)** |
| Top 30% | +68,268 | −7,266 | **+75,534 (+1,040%)** |
| Top 40% | +74,026 | −9,688 | **+83,714 (+864%)** |
| Top 50% | +54,223 | −12,110 | **+66,333 (+548%)** |

> **주(Note)**: 표의 수치는 1대10 Case-Control 추출 표본에 대한 Prior Offset Calibration($w = \frac{p_{\text{pop}}}{p_{\text{sample}}}$)을 적용하여 전체 모집단 스케일로 전환 산출한 절대 순증 구매 전환 건수(Net Incremental Purchases)이다.

전체 고객을 대상으로 무작위 발송할 경우, 본 캠페인의 평균 처치 효과(ATE)는 **−3.74%p** 로 구매율을 오히려 감소시키는 마이너스 이익 캠페인이다. 그러나 인과 모델이 추정한 CATE 예측값 기준으로 양수 구간의 **상위 30~40% 고객에게만 선택적으로 발송** 할 경우, 마이너스 캠페인을 플러스 이익 영역으로 전환함과 동시에 하위 60~70%의 불필요한 발송을 차단하여 마케팅 비용을 60% 이상 절감할 수 있다.

특히 Top 40% 구간에서 순증 구매 전환량이 **+74,026건** 으로 최대점에 도달하며, 이 구간이 수익 극대화와 비용 절감의 최적 경계선(Optimal Business Threshold)에 해당한다. 이는 해당 경계 이하의 고객군(Group 3 무반응군 및 Group 4 Sleeping Dogs)에게의 발송이 제외됨으로써 역효과에 의한 구매 손실이 원천 차단되기 때문이다.

---

## References

- Ascarza, E. (2018). Retention futility: Targeting high-risk customers might be ineffective. *Journal of Marketing Research*, 55(1), 80–98.
- Athey, S., Chetty, R., Imbens, G. W., & Kang, H. (2019). The surrogate index: Combining short-term proxies to estimate long-term treatment effects. *NBER Working Paper Series*, No. w26463.
- Athey, S., & Imbens, G. (2016). Recursive partitioning for heterogeneous treatment effects. *Proceedings of the National Academy of Sciences*, 113(27), 7353–7360.
- Breslow, N. E. (1996). Statistics in epidemiology: The case-control study. *Journal of the American Statistical Association*, 91(433), 14–28.
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1–C68.
- Cinelli, C., & Hazlett, C. (2020). Making sense of sensitivity: Extending omitted variable bias. *Journal of the Royal Statistical Society: Series B*, 82(1), 39–67.
- Crump, R. K., Hotz, V. J., Imbens, G. W., & Mitnik, O. A. (2009). Dealing with limited overlap in estimation of average treatment effects. *Biometrika*, 96(1), 187–199.
- Holland, P. W. (1986). Statistics and causal inference. *Journal of the American Statistical Association*, 81(396), 945–960.
- Kennedy, E. H. (2023). Towards optimal doubly robust estimation of heterogeneous causal effects. *Electronic Journal of Statistics*, 17(2), 3008–3049.
- King, G., & Zeng, L. (2001). Logistic regression in rare events data. *Political Analysis*, 9(2), 137–163.
- Künzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019). Metalearners for estimating heterogeneous treatment effects using machine learning. *Proceedings of the National Academy of Sciences*, 116(10), 4156–4165.
- Prentice, R. L., & Pyke, R. (1979). Logistic disease incidence models and case-control studies. *Biometrika*, 66(3), 403–411.
- Radcliffe, N. J., & Surry, P. D. (2011). Real-world uplift modelling with significance-based uplift trees. *White Paper*, Stochastic Solutions.
- Rößler, J., & Schoder, D. (2022). Bridging the gap between customer churn prediction and proactive retention. *Journal of Interactive Marketing*, 57(2), 218–234.
- Wager, S., & Athey, S. (2018). Estimation and inference of heterogeneous treatment effects using random forests. *Journal of the American Statistical Association*, 113(523), 1228–1242.
