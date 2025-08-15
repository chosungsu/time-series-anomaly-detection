"""
MTL(Multi-Task Learning) 모델 기반 실시간 이상 탐지 Streamlit 앱
- MTL 모델 자동 학습 및 저장
- 실시간 이상치 탐지 및 시각화
- 1초마다 자동 데이터 분석
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import os
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from models.mtl_learning.mtl_pipeline import ImprovedMultiTaskAnomalyDetector
from models.utils.data_util import load_data, preprocess_data

# 페이지 설정
st.set_page_config(
    page_title="실시간 이상 탐지",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .anomaly-detected {
        background-color: #ffebee;
        border-left-color: #f44336;
    }
    .normal-detected {
        background-color: #e8f5e8;
        border-left-color: #4caf50;
    }
    .training-status {
        background-color: #fff3e0;
        border-left-color: #ff9800;
    }
</style>
""", unsafe_allow_html=True)

# 전역 변수
MODEL_SAVE_DIR = "models/saved_model"
DATA_PATH = "data/sensor_data_rms2_fixed.csv"

@st.cache_resource
def ensure_model_directory():
    """모델 저장 디렉토리 생성"""
    Path(MODEL_SAVE_DIR).mkdir(parents=True, exist_ok=True)
    return MODEL_SAVE_DIR

@st.cache_resource
def load_or_train_model():
    """모델 로드 또는 학습 (캐싱)"""
    model_dir = ensure_model_directory()
    model_path = os.path.join(model_dir, "mtl_model.pkl")
    scaler_path = os.path.join(model_dir, "mtl_scaler.pkl")

    # 모델이 이미 존재하는지 확인
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            # 모델 로드
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # 스케일러 로드
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            
            # 모델 유효성 검사
            if hasattr(model, 'detect_anomalies') and callable(model.detect_anomalies):
                return model, scaler, True
            else:
                raise ValueError("Invalid model format")
                
        except Exception as e:
            st.warning(f"저장된 모델 로드 실패: {str(e)}")
            # 로드 실패 시 파일 삭제
            try:
                os.remove(model_path)
                os.remove(scaler_path)
                st.info("로드 실패한 모델 파일을 삭제했습니다.")
            except:
                pass
    
    # 모델이 없으면 학습
    return None, None, False

def train_mtl_model():
    """MTL 모델 학습"""
    st.info("🚀 MTL(Multi-Task Learning) 모델 학습을 시작합니다...")
    
    try:
        # 데이터 로드
        with st.spinner("데이터 로딩 중..."):
            data = load_data(DATA_PATH)
            if data is None:
                st.error("데이터를 로드할 수 없습니다.")
                return None, None
        
        # 데이터 전처리
        with st.spinner("데이터 전처리 중..."):
            data = preprocess_data(data)
        
        # 모델 초기화
        with st.spinner("MTL 모델 초기화 중..."):
            model = ImprovedMultiTaskAnomalyDetector(
                input_dim=data.shape[1],
                hidden_dim=64,
                latent_dim=32,
                window_size=7,
                epochs=50,
                device="cpu"
            )
        
        # 모델 학습
        with st.spinner("MTL 모델 학습 중... (Teacher 모델 학습 및 MTL 학습이 진행됩니다)"):
            model.fit(data)
        
        # 모델 저장
        model_dir = ensure_model_directory()
        model_path = os.path.join(model_dir, "mtl_model.pkl")
        scaler_path = os.path.join(model_dir, "mtl_scaler.pkl")
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(model.scaler, f)
        
        st.success("✅ MTL 모델 학습 및 저장이 완료되었습니다!")
        return model, model.scaler
        
    except Exception as e:
        st.error(f"모델 학습 중 오류 발생: {str(e)}")
        return None, None

def get_next_sensor_data():
    """실제 센서 데이터에서 순차적으로 데이터 포인트 추출 (실시간 시뮬레이션용)"""
    try:
        # 전역 변수로 실제 센서 데이터와 인덱스 관리
        if 'sensor_data_index' not in st.session_state:
            # 실제 센서 데이터 로드
            try:
                sensor_data = load_data(DATA_PATH)
                if sensor_data is not None:
                                    sensor_data = preprocess_data(sensor_data)
                st.session_state.sensor_data = sensor_data
                st.session_state.sensor_data_index = 0
                st.session_state.sensor_data_length = len(sensor_data)
                
                # 전역 통계 정보 계산 및 저장 (가중치 1.5배 적용을 위한 기반)
                st.session_state.global_stats = {
                    'mean': sensor_data.mean().values,
                    'std': sensor_data.std().values,
                    'min': sensor_data.min().values,
                    'max': sensor_data.max().values,
                    'range': (sensor_data.max() - sensor_data.min()).values
                }

            except Exception as e:
                st.warning(f"센서 데이터 로드 실패: {str(e)}")
                return np.random.normal(0, 0.1, 7).astype(np.float32)
        
        # 실제 센서 데이터에서 순차적으로 데이터 포인트 추출
        if hasattr(st.session_state, 'sensor_data') and st.session_state.sensor_data is not None:
            current_index = st.session_state.sensor_data_index
            
            # 인덱스가 데이터 길이를 초과하면 처음부터 다시 시작 (순환)
            if current_index >= st.session_state.sensor_data_length:
                st.session_state.sensor_data_index = 0
                current_index = 0
            
            # 현재 인덱스의 데이터 포인트 추출
            data_point = st.session_state.sensor_data.iloc[current_index].values
            
            # 다음 인덱스로 이동
            st.session_state.sensor_data_index += 1
            
            # 데이터 유효성 검사
            if data_point is None or len(data_point) != 7 or np.any(np.isnan(data_point)):
                # 기본값으로 대체
                data_point = np.random.normal(0, 0.1, 7)
            
            return data_point.astype(np.float32)
        else:
            # 센서 데이터가 없는 경우 기본값 반환
            return np.random.normal(0, 0.1, 7).astype(np.float32)
            
    except Exception as e:
        # 오류 발생 시 기본값 반환
        return np.random.normal(0, 0.1, 7).astype(np.float32)

def create_window_from_data_point(data_point):
    """데이터 포인트를 7개 feature로 구성된 윈도우로 변환"""
    # 데이터 포인트는 이미 7개 feature를 가지고 있음
    # 이를 1개 윈도우로 구성 (1, 7) 형태
    return data_point.reshape(1, -1)

def get_anomaly_color(score, threshold_95, threshold_75):
    """이상 점수에 따른 색상 반환"""
    try:
        if score >= threshold_95:
            return 'red'      # 95% 이상: 빨간색
        elif score >= threshold_75:
            return 'yellow'   # 75% 이상: 노란색
        else:
            return 'blue'     # 정상: 파란색
    except:
        return 'blue'  # 오류 시 기본값

def main():
    """메인 함수"""
    st.markdown('<h1 class="main-header">🔍 MTL 실시간 이상 탐지</h1>', unsafe_allow_html=True)
    
    # 사이드바
    st.sidebar.title("⚙️ 설정")
    
    # 모델 로드 또는 학습
    model, scaler, model_loaded = load_or_train_model()
    
    if not model_loaded:
        st.sidebar.warning("⚠️ 저장된 모델이 없습니다.")
        
        if st.sidebar.button("🚀 모델 학습 시작", type="primary"):
            model, scaler = train_mtl_model()
            if model is not None:
                model_loaded = True
                st.rerun()
    
    if not model_loaded:
        st.error("모델을 로드하거나 학습할 수 없습니다.")
        st.stop()
    
    st.sidebar.success("✅ 모델 로드 완료")
    
        # 임계값 조정 (가중치 1.5배 적용 후 범위에 맞게 조정)
    threshold_95 = st.sidebar.slider(
        "95% 이상치 임계값 (심각)",
        min_value=0.0,
        max_value=2.0,
        value=0.75,  # 0.5 * 1.5 = 0.75
        step=0.1,
        help="이 값보다 높은 이상 점수를 95% 이상 이상으로 판단합니다. (가중치 1.5배 적용 후: 0.75 이상 심각 이상치)"
    )
    
    threshold_75 = st.sidebar.slider(
        "75% 이상치 임계값 (주의)",
        min_value=0.0,
        max_value=2.0,
        value=0.45,  # 0.3 * 1.5 = 0.45
        step=0.1,
        help="이 값보다 높은 이상 점수를 75% 이상 이상으로 판단합니다. (가중치 1.5배 적용 후: 0.45 이상 주의 이상치)"
    )
    
    # 사이드바에 구분선 추가
    st.sidebar.divider()
    
    # 실시간 이상치 탐지 시작/중지 버튼 (사이드바에 배치)
    if 'detection_running' not in st.session_state:
        st.session_state.detection_running = False
    
    # 슬라이딩 윈도우를 위한 데이터 구조 변경
    if 'anomaly_data' not in st.session_state:
        st.session_state.anomaly_data = {
            'timestamps': [],
            'scores': [],
            'colors': [],
            'window_sequences': [],  # 7개 feature로 구성된 윈도우 시퀀스들
            'raw_data': [],  # 원본 데이터 포인트들 (시각화용)
            'anomaly_scores': []  # 이상 점수들 (시각화용)
        }
    
    # 사이드바에 이상치 탐지 버튼 배치
    if not st.session_state.detection_running:
        if st.sidebar.button("🚀 이상치 탐지 시작", type="primary", use_container_width=True):
            st.session_state.detection_running = True
            st.session_state.anomaly_data = {
                'timestamps': [],
                'scores': [],
                'colors': [],
                'window_sequences': [],
                'raw_data': [],
                'anomaly_scores': []
            }
            st.rerun()
    else:
        if st.sidebar.button("⏹️ 이상치 탐지 중지", type="secondary", use_container_width=True):
            st.session_state.detection_running = False
            st.rerun()
    
    # 사이드바에 상태 표시
    if st.session_state.detection_running:
        st.sidebar.success("🟢 실시간 이상치 탐지가 실행 중입니다")
    else:
        st.sidebar.info("⏸️ 이상치 탐지가 중지되었습니다")

    # 최근 결과 요약 (차트 위에 표시)
    if st.session_state.anomaly_data['scores']:
        st.subheader("📊 최근 분석 결과 요약")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("총 분석 포인트", len(st.session_state.anomaly_data['scores']))
        
        with col2:
            red_count = st.session_state.anomaly_data['colors'].count('red')
            st.metric("95% 이상 이상 (0.75+)", red_count, delta=f"{red_count/len(st.session_state.anomaly_data['scores'])*100:.1f}%")
        
        with col3:
            yellow_count = st.session_state.anomaly_data['colors'].count('yellow')
            st.metric("75% 이상 이상 (0.45+)", yellow_count, delta=f"{yellow_count/len(st.session_state.anomaly_data['scores'])*100:.1f}%")
        
        with col4:
            blue_count = st.session_state.anomaly_data['colors'].count('blue')
            st.metric("정상", blue_count, delta=f"{blue_count/len(st.session_state.anomaly_data['scores'])*100:.1f}%")
    
    # 실시간 차트
    if st.session_state.detection_running:
        # 차트 컨테이너
        chart_container = st.container()
        
        with chart_container:
            # 실시간 업데이트를 위한 placeholder
            chart_placeholder = st.empty()
            
            # 실시간 데이터 수집 및 차트 업데이트
            try:
                # 실제 센서 데이터에서 다음 데이터 포인트 추출
                data_point = get_next_sensor_data()
                
                # 데이터 유효성 검사
                if data_point is None or len(data_point) != 7:
                    st.error("데이터 생성에 실패했습니다.")
                    return
                
                # 현재 시간
                current_time = time.strftime("%H:%M:%S")
                
                # 슬라이딩 윈도우 방식: 7개 윈도우가 필요함
                if len(st.session_state.anomaly_data['window_sequences']) < 7:
                    # 윈도우 시퀀스가 7개 미만이면 예측을 수행하지 않음
                    # 대신 정상으로 처리하고 데이터를 계속 수집
                    score = 0.0  # MTL 모델의 정상 상태 전형적인 점수 (0에 가까움)
                    color = 'blue'
                    anomaly_score = 0.0  # 이상 점수도 0으로 설정
                else:
                    # 슬라이딩 윈도우 구성: 마지막 7개 윈도우 사용
                    # MTL 모델은 window_size=7을 기대함
                    last_7_windows = st.session_state.anomaly_data['window_sequences'][-7:]
                    
                    # 7개 윈도우를 하나의 DataFrame으로 결합
                    # 각 윈도우는 7개 feature를 가짐
                    all_windows = []
                    for i, window in enumerate(last_7_windows):
                        # window는 (1, 7) 형태이므로 flatten()으로 (7,)로 변환
                        window_data = window.flatten()
                        all_windows.append(window_data)
                    
                    # 7개 윈도우를 DataFrame으로 변환 (7개 행, 7개 열)
                    input_df = pd.DataFrame(
                        all_windows,
                        columns=[
                            'Pressure', 'Power1', 'Power2', 'Vibration_Peak1', 
                            'Vibration_RMS1', 'Vibration_Peak2', 'Vibration_RMS2'
                        ]
                    )
                    
                    try:
                        # MTL 모델로 이상치 탐지
                        # MTL 모델은 score_samples 메서드를 사용
                        scores = model.score_samples(input_df)
                        
                        # 결과 유효성 검사
                        if scores is None or len(scores) == 0:
                            st.error("모델 예측 결과가 비어있습니다.")
                            return
                        
                        # scores는 윈도우 단위로 계산된 점수
                        # 마지막 윈도우의 점수를 사용 (현재 시점)
                        score = float(scores[-1])
                        
                        # 가중치 1.5배 적용하여 출력 범위 확장
                        score = score * 1.5
                        anomaly_score = score  # 이상 점수 저장
                        
                        color = get_anomaly_color(score, threshold_95, threshold_75)
                        
                    except Exception as model_error:
                        st.warning(f"모델 예측 중 오류 발생: {str(model_error)}")
                        # 대안: 간단한 통계적 이상치 탐지
                        data_mean = np.mean(data_point)
                        data_std = np.std(data_point)
                        z_scores = np.abs((data_point - data_mean) / (data_std + 1e-8))
                        max_z_score = np.max(z_scores)
                        
                        # Z-score 기반 이상치 점수
                        if max_z_score > 3.0:
                            score = max_z_score / 3.0  # 정규화
                        else:
                            score = 0.0  # MTL 모델의 정상 상태 점수
                        
                        # 가중치 1.5배 적용하여 출력 범위 확장
                        score = score * 1.5
                        anomaly_score = score  # 이상 점수 저장
                        color = get_anomaly_color(score, threshold_95, threshold_75)
                
                # 데이터 저장
                st.session_state.anomaly_data['timestamps'].append(current_time)
                st.session_state.anomaly_data['scores'].append(score)
                st.session_state.anomaly_data['colors'].append(color)
                st.session_state.anomaly_data['raw_data'].append(data_point.copy())  # 원본 데이터 저장
                st.session_state.anomaly_data['anomaly_scores'].append(anomaly_score)  # 이상 점수 저장
                
                # 현재 데이터 포인트를 윈도우 시퀀스로 변환하여 저장
                current_window = create_window_from_data_point(data_point)
                st.session_state.anomaly_data['window_sequences'].append(current_window)
                
                # 최근 7개 윈도우 시퀀스만 유지 (슬라이딩 윈도우)
                if len(st.session_state.anomaly_data['window_sequences']) > 7:
                    st.session_state.anomaly_data['window_sequences'] = st.session_state.anomaly_data['window_sequences'][-7:]
                
                # MTL 모델의 경우 초기 7개 데이터 포인트는 0.0으로 설정
                if len(st.session_state.anomaly_data['scores']) <= 7:
                    score = 0.0
                    color = 'blue'
                    anomaly_score = 0.0
                
                # 데이터 제한 제거 - 모든 데이터 유지
                # if len(st.session_state.anomaly_data['timestamps']) > 100:
                #     st.session_state.anomaly_data['timestamps'] = st.session_state.anomaly_data['timestamps'][-100:]
                #     st.session_state.anomaly_data['scores'] = st.session_state.anomaly_data['scores'][-100:]
                #     st.session_state.anomaly_data['colors'] = st.session_state.anomaly_data['colors'][-100:]
                
                # 이상치 점수만 표시하는 단순한 차트 생성
                fig = go.Figure()
                
                # 이상 점수 라인 (마커와 함께)
                if len(st.session_state.anomaly_data['anomaly_scores']) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=st.session_state.anomaly_data['timestamps'],
                            y=st.session_state.anomaly_data['anomaly_scores'],
                            mode='lines+markers',
                            name='이상 점수',
                            line=dict(color='#1f77b4', width=3),
                            marker=dict(
                                size=8,
                                color=st.session_state.anomaly_data['colors'],
                                colorscale='Viridis'
                            )
                        )
                    )
                
                # 임계값 라인들
                # 95% 임계값 라인 (빨간색 점선)
                fig.add_hline(
                    y=threshold_95,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"95% 임계값: {threshold_95:.1f}",
                    annotation_position="top right"
                )
                
                # 75% 임계값 라인 (주황색 점선)
                fig.add_hline(
                    y=threshold_75,
                    line_dash="dash",
                    line_color="orange",
                    annotation_text=f"75% 임계값: {threshold_75:.1f}",
                    annotation_position="top right"
                )
                
                # 차트 레이아웃 설정
                fig.update_layout(
                    title="MTL 실시간 이상치 탐지 결과",
                    xaxis_title="시간",
                    yaxis_title="이상 점수",
                    height=600,
                    showlegend=True,
                    xaxis=dict(tickangle=45),
                    yaxis=dict(
                        range=[0, 1.5],  # 가중치 1.5배 적용 후 예상 범위 (0.8 * 1.5 = 1.2)
                        title="이상 점수"
                    )
                )
                
                # 차트 업데이트
                chart_placeholder.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"이상 탐지 중 오류 발생: {str(e)}")
                st.error("데이터 형태를 확인해주세요.")
                return
    
    # 실시간 업데이트를 위한 자동 새로고침
    if st.session_state.detection_running:
        time.sleep(1)
        st.rerun()

if __name__ == "__main__":
    main()
