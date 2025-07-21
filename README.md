# 🔍 실시간 이상 탐지 시스템 (PPO)

<h1 align="center">TEAM 초코pytorch - 실시간 이상 탐지 시스템</h1>

## 📋 목차

- [프로젝트 소개](#프로젝트-소개)
- [주요 기능](#주요-기능)
- [기술 스택](#기술-스택)
- [프로젝트 구조](#프로젝트-구조)
- [설치 및 실행](#설치-및-실행)
- [사용법](#사용법)
- [성능 지표](#성능-지표)
- [기여자](#기여자)

## 🎯 프로젝트 소개

이 프로젝트는 제조 공정 장비에서 실시간으로 수집되는 시계열 데이터에 대한 이상 탐지 및 예측 모델을 개발한 프로젝트입니다. 

**주요 특징:**
- 🤖 **PPO (Proximal Policy Optimization)** 기반 강화학습 모델
- 📊 **실시간 데이터 시각화** 및 모니터링
- ⚡ **순차적 데이터 처리** (인덱스 0부터 순차 진행)
- 🎛️ **수동/자동 업데이트** 모드 지원
- 📈 **실시간 성능 지표** 모니터링

## ✨ 주요 기능

### 1. 실시간 이상 탐지
- PPO 강화학습 모델을 활용한 실시간 이상 탐지
- 순차적 데이터 처리로 안정적인 탐지 성능
- 실시간 차트 업데이트로 현재 처리 상황 시각화

### 2. 대화형 웹 인터페이스
- Streamlit 기반 직관적인 사용자 인터페이스
- 실시간 차트 및 성능 지표 표시
- 시작/정지/리셋 버튼으로 쉬운 제어

### 3. 성능 모니터링
- 정확도, 정밀도, 재현율, F1 점수 실시간 계산
- 이상 탐지 목록 및 상세 정보 표시
- 처리 진행률 및 현재 상태 모니터링

### 4. 유연한 업데이트 모드
- **자동 업데이트**: 설정 가능한 간격으로 자동 진행
- **수동 업데이트**: 버튼 클릭으로 단계별 진행 (스크롤 문제 해결)

## 🛠️ 기술 스택

### Backend & Algorithm
- **Python 3.8+**
- **PyTorch**: PPO 모델 구현
- **Streamlit**: 웹 애플리케이션 프레임워크
- **Plotly**: 실시간 데이터 시각화
- **Pandas**: 데이터 처리 및 분석
- **NumPy**: 수치 계산

### Machine Learning
- **PPO (Proximal Policy Optimization)**: 강화학습 기반 이상 탐지
- **Custom Environment**: 이상 탐지를 위한 평가 환경
- **Real-time Processing**: 순차적 데이터 처리

### Data Visualization
- **Plotly Graph Objects**: 실시간 차트 생성
- **Interactive Charts**: 현재 처리 포인트 및 이상 탐지 표시
- **Responsive Design**: 다양한 화면 크기 지원

## 📁 프로젝트 구조

```
bistelligence/
├── meta_aad/                       # 강화학습 모델
│   ├── env.py                      # 환경 정의
│   ├── ppo2.py                     # PPO 모델 구현
│   ├── agents.py                   # 에이전트 정의
│   └── utils.py                    # 유틸리티 함수
│
├── data/                           # 데이터 파일
│   └── sensor_data_with_anomalylabel_isolationforest.csv
│
├── log/                            # 학습 로그
│   └── model.pth                   # 학습된 PPO 모델
│
├── results/                        # 결과 파일
│
├── src/                            # 소스 코드
│   └── img/                        # 이미지 파일
│
├── util/                           # 유틸리티
│   ├── eda.py                      # 탐색적 데이터 분석
│   └── preprocess.py               # 데이터 전처리
│
├── app.py                          # 메인 Streamlit 애플리케이션
├── evaluate.py                     # 모델 평가 스크립트
├── train.py                        # 모델 학습 스크립트
├── requirements.txt                # Python 의존성
└── README.md                       # 프로젝트 문서
```

## 🚀 설치 및 실행

### 1. 저장소 클론
```bash
git clone https://github.com/ChocoPytorch/BISTelligence.git
cd BISTelligence
```

### 2. 의존성 설치
```bash
pip install -r requirements.txt
```

### 3. 애플리케이션 실행
```bash
streamlit run app.py
```

### 4. 브라우저에서 접속
```
http://localhost:8501
```

## 📖 사용법

### 1. 애플리케이션 시작
- 브라우저에서 `http://localhost:8501` 접속
- 데이터 로드 및 모델 초기화 완료 확인

### 2. 설정 조정
- **자동 업데이트**: 체크박스로 자동/수동 모드 선택
- **업데이트 간격**: 자동 모드에서 처리 간격 설정 (0.1~2.0초)
- **차트 윈도우 크기**: 차트에 표시할 데이터 포인트 수 (50~200)

### 3. 이상 탐지 시작
- **▶️ 시작**: 실시간 이상 탐지 시작
- **⏸️ 정지**: 탐지 중단
- **🔄 리셋**: 모든 상태 초기화

### 4. 결과 확인
- **실시간 차트**: 현재 처리 중인 데이터 포인트 표시
- **성능 지표**: 정확도, 정밀도, 재현율, F1 점수
- **이상 탐지 목록**: 탐지된 이상 데이터 상세 정보

## 📊 성능 지표

### 실시간 모니터링
- **정확도 (Accuracy)**: 전체 예측 중 올바른 예측 비율
- **정밀도 (Precision)**: 이상으로 탐지된 것 중 실제 이상 비율
- **재현율 (Recall)**: 실제 이상 중 탐지된 이상 비율
- **F1 점수**: 정밀도와 재현율의 조화평균

### 시각화
- **실시간 차트**: 현재 처리 중인 데이터 포인트 하이라이트
- **이상 탐지 표시**: 탐지된 이상 포인트를 빨간색 X로 표시
- **진행률**: 전체 데이터 대비 처리 완료 비율

## 🔧 주요 개선사항

### 최근 업데이트
- ✅ **순차적 데이터 처리**: 인덱스 0부터 순차적으로 처리
- ✅ **실시간 차트 개선**: 현재 처리 포인트 시각화
- ✅ **스크롤 문제 해결**: 수동 업데이트 모드 추가
- ✅ **성능 지표 표시**: 실시간 성능 모니터링
- ✅ **이상 탐지 목록**: 탐지된 이상만 별도 표시

### 기술적 개선
- **PPO 모델 통합**: 강화학습 기반 이상 탐지
- **환경 기반 평가**: `EvalEnv`를 활용한 정확한 평가
- **예외 처리 강화**: 안정적인 애플리케이션 실행
- **UI/UX 개선**: 직관적인 사용자 인터페이스

## 👥 기여자

이 프로젝트에 참여한 팀원들입니다.

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

---

**BISTelligence.ai** - 제조 공정 이상 탐지 솔루션

