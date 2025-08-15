# MTL(Multi-Task Learning) 기반 실시간 시계열 이상 탐지 시스템

## 🎯 프로젝트 개요

이 프로젝트는 **MTL(Multi-Task Learning)** 아키텍처를 사용한 실시간 시계열 이상 탐지 시스템입니다. **Anomaly Transformer**와 **OmniAnomaly**를 Teacher 모델로 활용하여 지식 증류를 통해 효율적인 이상 탐지를 수행합니다.

### 주요 특징

- **MTL 아키텍처**: 복원(Reconstruction), 예측(Forecasting), 분류(Classification) 작업을 동시에 학습
- **Teacher-Student 지식 증류**: Anomaly Transformer와 OmniAnomaly의 지식을 Student 모델에 전이
- **실시간 단일 윈도우 추론**: 7개 타임스텝 × 7개 특징의 단일 윈도우에 대한 즉시 추론
- **다중 스케일 특징 추출**: CNN 기반 다중 스케일 인코더와 BiLSTM을 통한 시계열 특징 학습
- **가중치 1.5배 적용**: 출력 범위 확장을 통한 더 정밀한 이상 탐지

## 프로젝트 구조

```
bistelligence/
├── models/                      # 다양한 이상 탐지 모델들
│   ├── mtl_learning/           # MTL 모델 (핵심)
│   │   ├── mtl_pipeline.py    # ImprovedMTL 및 ImprovedMultiTaskAnomalyDetector
│   │   └── __init__.py
│   ├── an_transformer/         # Anomaly Transformer (Teacher)
│   │   ├── anomaly_transformer_model.py
│   │   └── __init__.py
│   ├── omnianomaly/            # OmniAnomaly (Teacher)
│   │   ├── omnianomaly_model.py
│   │   └── __init__.py
│   ├── deep_svdd/              # Deep SVDD 모델
│   ├── isolation_forest/       # Isolation Forest 모델
│   ├── lof/                    # Local Outlier Factor 모델
│   ├── oc_svm/                 # One-Class SVM 모델
│   ├── thoc/                   # THOC 모델
│   └── utils/                  # 공통 유틸리티
│       ├── data_util.py        # 데이터 로딩 및 전처리
│       └── unsupervised_evaluator.py  # 비지도 학습 평가
├── app.py                      # Streamlit 실시간 이상 탐지 앱
├── test.py                     # 모든 모델의 성능 평가
├── data/                       # 센서 데이터
│   └── sensor_data_rms2_fixed.csv
├── leaderboard/                # 모델 성능 비교 결과
├── requirements.txt             # 의존성 패키지
└── README.md                   # 이 파일
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 또는
.venv\Scripts\activate     # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. 모델 평가

```bash
# 개별 모델 평가
python test.py
```

### 3. Streamlit 앱 실행

```bash
# 실시간 이상 탐지 웹 애플리케이션
streamlit run app.py
```

## 📱 Streamlit 웹 애플리케이션

### 주요 기능

1. **🚀 실시간 이상 탐지**
   - MTL 모델 자동 학습 및 저장
   - 1초마다 자동 데이터 분석
   - 실제 센서 데이터 기반 실시간 추론

2. **📊 동적 시각화**
   - 실시간 이상 점수 차트
   - 임계값 기반 색상 코딩 (빨강: 심각, 노랑: 주의, 파랑: 정상)
   - 분석 결과 요약 메트릭

3. **⚙️ 실시간 설정**
   - 95% 이상치 임계값 조정 (0.75+)
   - 75% 이상치 임계값 조정 (0.45+)
   - 가중치 1.5배 적용으로 출력 범위 확장

4. **🔬 모델 관리**
   - 자동 모델 학습 및 저장
   - 모델 상태 모니터링
   - 오류 처리 및 복구

### 사용법

1. **모델 로드**: 앱 시작 시 자동으로 학습된 MTL 모델을 로드
2. **이상 탐지 시작**: 사이드바의 "🚀 이상치 탐지 시작" 버튼 클릭
3. **임계값 조정**: 실시간으로 이상 탐지 기준점 조정
4. **결과 모니터링**: 차트와 메트릭을 통한 실시간 이상 탐지 결과 확인

## 🔧 핵심 모델 아키텍처

### MTL(Multi-Task Learning) 모델

#### **1. 다중 스케일 CNN 인코더**
- **3개 CNN 블록**: 채널 16→32→32, 커널=3, dilation=1/2/4
- **병목 특징**: 32차원 잠재 공간으로 특징 압축
- **노이즈 주입**: 0.05~0.1 가우시안 노이즈로 DAE 학습

#### **2. 공유 BiLSTM 인코더**
- **양방향 LSTM**: 시계열의 순차적 의존성 학습
- **1개 레이어**: 효율성과 성능의 균형
- **잠재 공간 변환**: 32차원 특징을 16차원으로 압축

#### **3. 다중 작업 헤드**
- **복원 헤드**: 입력 시계열 재구성
- **예측 헤드**: 다음 시점 예측
- **분류 헤드**: 이상/정상 이진 분류

### Teacher 모델들

#### **Anomaly Transformer**
- **Association Discrepancy**: 정상 패턴과의 연관성 차이 기반 이상 탐지
- **Self-Attention**: 시계열 내 장거리 의존성 학습
- **통계적 특징**: 17개 통계적 특징으로 입력 확장

#### **OmniAnomaly**
- **VAE 기반**: 변분 오토인코더를 통한 정상 패턴 학습
- **Stochastic RNN**: 확률적 순환 신경망으로 불확실성 모델링
- **재구성 오차**: 입력과 출력의 차이로 이상 점수 계산

## 📊 이상 점수 계산

이상 점수는 다음 구성 요소들의 가중 합으로 계산됩니다:

```
Score = α × Reconstruction_Error + β × Forecasting_Error + γ × Classification_Score
```

- **α (reconstruction_weight)**: 재구성 오차 가중치 (0.4)
- **β (forecasting_weight)**: 예측 오차 가중치 (0.3)  
- **γ (classification_weight)**: 분류 점수 가중치 (0.3)

### **가중치 1.5배 적용**
- **출력 범위 확장**: 0~0.8 → 0~1.2
- **임계값 조정**: 95% (0.75+), 75% (0.45+)
- **전역 통계 활용**: 데이터셋 전체 통계 정보 기반 스케일링

## ⚙️ 설정 및 튜닝

### 모델 매개변수
- **윈도우 크기**: 7 (Teacher 모델과 호환)
- **은닉 차원**: 64 (MTL 모델)
- **잠재 차원**: 32 (병목 특징)
- **학습 에포크**: 50 (효율성과 성능의 균형)
- **배치 크기**: 64 (메모리와 학습 안정성 고려)

### 실시간 조정
- **95% 임계값**: 0.75 이상 (심각 이상치)
- **75% 임계값**: 0.45 이상 (주의 이상치)
- **가중치**: 1.5배 적용으로 출력 범위 확장
- **모니터링**: 실시간 성능 지표 확인

## 📈 성능 지표

- **정확도**: 전체 예측 중 정확한 예측의 비율
- **정밀도**: 이상으로 예측한 것 중 실제 이상의 비율
- **재현율**: 실제 이상 중 이상으로 예측된 비율
- **F1 점수**: 정밀도와 재현율의 조화평균
- **ROC AUC**: 수신자 조작 특성 곡선 아래 면적

## 🚨 주의사항

1. **데이터 전처리**: 입력 데이터는 정규화되어야 함
2. **윈도우 크기**: 정확히 7 타임스텝이어야 함 (학습된 모델과 호환)
3. **메모리 관리**: GPU 사용 시 배치 크기 조정 필요
4. **임계값 설정**: 가중치 1.5배 적용 후 조정된 임계값 사용

## 📚 참고 자료

### 핵심 논문
- **[ANOMALY TRANSFORMER: TIME SERIES ANOMALY DETECTION WITH ASSOCIATION DISCREPANCY](https://arxiv.org/abs/2110.02642)**: Anomaly Transformer의 Association Discrepancy 기법
- **[ROBUST ANOMALY DETECTION FOR MULTIVARIATE TIME SERIES THROUGH STOCHASTIC RECURRENT NEURAL NETWORK](https://netman.aiops.org/wp-content/uploads/2019/08/OmniAnomaly_camera-ready.pdf)**: OmniAnomaly의 확률적 RNN 기반 이상 탐지
- **[AN EFFICIENT FEDERATED DISTILLATION LEARNING SYSTEM FOR MULTI-TASK TIME SERIES CLASSIFICATION](https://arxiv.org/abs/2201.00011)**: 다중 작업 시계열 분류를 위한 연합 증류 학습

## 👥 Contributors

Team members who participated in this project.

<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Chocopytorch"><img src="https://avatars.githubusercontent.com/u/122209595?v=4?s=100" width="100px;" alt="Chocopytorch"/><br /><sub><b>Chocopytorch</b></sub></a><br /><a href="https://github.com/Chocopytorch/BISTelligence/commits?author=Chocopytorch" title="Commits">📖</a> </td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/chosungsu"><img src="https://avatars.githubusercontent.com/u/48382347?v=4?s=100" width="100px;" alt="chosungsu"/><br /><sub><b>chosungsu</b></sub></a><br /><a href="https://github.com/Chocopytorch/BISTelligence/commits?author=chosungsu" title="Commits">📖</a> </td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/kmw4097"><img src="https://avatars.githubusercontent.com/u/98750892?v=4?s=100" width="100px;" alt="kmw4097"/><br /><sub><b>kmw4097</b></sub></a><br /><a href="https://github.com/Chocopytorch/BISTelligence/commits?author=kmw4097" title="Commits">📖</a> </td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/dbnub"><img src="https://avatars.githubusercontent.com/u/99518647?v=4?s=100" width="100px;" alt="dbnub"/><br /><sub><b>dbnub</b></sub></a><br /><a href="https://github.com/Chocopytorch/BISTelligence/commits?author=dbnub" title="Commits">📖</a> </td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/choiyongwoo"><img src="https://avatars.githubusercontent.com/u/50268222?v=4?s=100" width="100px;" alt="choiyongwoo"/><br /><sub><b>choiyongwoo</b></sub></a><br /><a href="https://github.com/Chocopytorch/BISTelligence/commits?author=choiyongwoo" title="Commits">📖</a> </td>
    </tr>
  </tbody>
</table>
