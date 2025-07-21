import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
import os
import torch
from meta_aad.ppo2 import PPO
from meta_aad.env import make_eval_env


# 페이지 설정
st.set_page_config(
    page_title="실시간 이상 탐지 시스템",
    page_icon="📊",
    layout="wide"
)

# 한글 폰트 설정
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .anomaly-alert {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
        color: #000000 !important;
    }
    .normal-log {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 4px;
        color: #000000 !important;
    }
    .performance-metrics {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
        color: #000000 !important;
    }
    .stMarkdown {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# 데이터 로드 함수
@st.cache_data
def load_sensor_data(d):
    """센서 데이터 로드"""
    if os.path.exists(d):
        df = pd.read_csv(d)
        return df
    else:
        st.error(f"데이터 파일을 찾을 수 없습니다: {d}")
        return None

# 순차적 처리를 위한 예측 함수
def predict_anomaly_sequential(model, env, data, current_index):
    """순차적으로 데이터를 처리하는 예측 함수"""
    try:
        # 현재 인덱스가 legal actions에 있는지 확인
        if current_index not in env.legal:
            # 이미 처리된 인덱스라면 다음 처리 가능한 인덱스 찾기
            remaining_legal = sorted(env.legal)
            if remaining_legal:
                current_index = remaining_legal[0]
            else:
                return False, 0.0, 1.0, 0, 0, current_index, 0.0, True
        
        # 현재 상태에서 모든 가능한 액션에 대한 observation 계산
        state_features = env._obs()
        
        # 현재 인덱스를 액션으로 선택 (순차적 처리)
        action = current_index
        
        # 액션을 취하기 전의 ground truth 저장 (evaluate_policy와 동일)
        ground_truth = env._Y[action]  # 0: anomaly, 1: normal
        
        # 환경에서 step 실행 (evaluate_policy와 동일)
        obs, legal, reward, done, _ = env.step(action)
        
        # 정책 네트워크로 액션 확률 계산 (디버깅용)
        action_features = state_features[action]
        state_tensor = torch.FloatTensor(action_features).unsqueeze(0).to(model.device)
        with torch.no_grad():
            action_probs = model.policy(state_tensor)
        
        # 액션 확률도 반환
        anomaly_prob = action_probs[0, 1].cpu().numpy().item()
        normal_prob = action_probs[0, 0].cpu().numpy().item()
        
        # evaluate_policy와 정확히 동일한 방식으로 이상 판단
        is_anomaly = (reward > 0)  # reward > 0이면 이상 탐지 성공
        
        # 선택된 액션의 데이터 값
        feature_cols = [col for col in data.columns if col not in ['key', 'anomaly_label', 'anomaly_score']]
        if 'Pressure' in feature_cols:
            selected_value = data.iloc[action]['Pressure']
        else:
            selected_value = data.iloc[action][feature_cols[0]] if feature_cols else action
        
        return is_anomaly, anomaly_prob, normal_prob, reward, ground_truth, action, selected_value, done
        
    except Exception as e:
        st.error(f"PPO 모델 예측 중 오류: {e}")
        import traceback
        st.error(traceback.format_exc())
        return False, 0.0, 1.0, 0, 0, 0, 0.0, True

# 실시간 데이터 차트 생성 함수
def create_realtime_chart(data, current_index, window_size=100, anomaly_indices=None, is_running=False):
    """실시간 데이터 차트 생성 (순차적 처리 기반)"""
    try:
        if not is_running:
            # 시작 전에는 빈 차트 표시
            start_idx = 0
            end_idx = min(window_size, len(data))
            window_data = data.iloc[start_idx:end_idx]
            
            # 차트 생성
            fig = go.Figure()
            
            # 특성 컬럼들 선택 (key, anomaly_label, anomaly_score 제외)
            feature_cols = [col for col in window_data.columns if col not in ['key', 'anomaly_label', 'anomaly_score']]
            
            # 첫 번째 특성으로 차트 생성 (Pressure 우선, 없으면 첫 번째 특성)
            if 'Pressure' in feature_cols:
                feature_col = 'Pressure'
            else:
                feature_col = feature_cols[0] if feature_cols else 'index'
            
            # 메인 데이터 라인만 표시 (처리 전)
            fig.add_trace(
                go.Scatter(
                    x=window_data.index,
                    y=window_data[feature_col] if feature_col != 'index' else window_data.index,
                    mode='lines',
                    name=f'{feature_col}',
                    line=dict(color='#1f77b4', width=2),
                    showlegend=True
                )
            )
            
            # 레이아웃 설정
            fig.update_layout(
                height=500,
                title=f'실시간 데이터 모니터링 - {feature_col} (시작 대기 중)',
                xaxis_title='데이터 인덱스',
                yaxis_title=feature_col,
                showlegend=True,
                hovermode='x unified'
            )
            
            # x축 범위 설정
            fig.update_xaxes(range=[start_idx, end_idx])
            
            return fig
        else:
            # 실행 중일 때는 현재 인덱스 기반으로 윈도우 설정
            start_idx = max(0, current_index - window_size + 1)
            end_idx = min(len(data), current_index + 1)
            window_data = data.iloc[start_idx:end_idx]
        
            # 차트 생성
            fig = go.Figure()
            
            # 특성 컬럼들 선택 (key, anomaly_label, anomaly_score 제외)
            feature_cols = [col for col in window_data.columns if col not in ['key', 'anomaly_label', 'anomaly_score']]
            
            # 첫 번째 특성으로 차트 생성 (Pressure 우선, 없으면 첫 번째 특성)
            if 'Pressure' in feature_cols:
                feature_col = 'Pressure'
            else:
                feature_col = feature_cols[0] if feature_cols else 'index'
            
            # 메인 데이터 라인
            fig.add_trace(
                go.Scatter(
                    x=window_data.index,
                    y=window_data[feature_col] if feature_col != 'index' else window_data.index,
                    mode='lines',
                    name=f'{feature_col}',
                    line=dict(color='#1f77b4', width=2),
                    showlegend=True
                )
            )
            
            # 현재 처리 중인 포인트 하이라이트
            if start_idx <= current_index <= end_idx:
                current_value = data.iloc[current_index][feature_col] if feature_col != 'index' else current_index
                fig.add_trace(
                    go.Scatter(
                        x=[current_index],
                        y=[current_value],
                        mode='markers',
                        marker=dict(color='red', size=10, symbol='diamond'),
                        name='현재 처리 포인트',
                        showlegend=True
                    )
                )
            
            # 이상 탐지된 포인트들 표시
            if anomaly_indices is not None:
                anomaly_values = []
                anomaly_x_indices = []
                for idx in anomaly_indices:
                    if start_idx <= idx <= end_idx:
                        anomaly_value = data.iloc[idx][feature_col] if feature_col != 'index' else idx
                        anomaly_values.append(anomaly_value)
                        anomaly_x_indices.append(idx)
                
                if anomaly_values:
                    fig.add_trace(
                        go.Scatter(
                            x=anomaly_x_indices,
                            y=anomaly_values,
                            mode='markers',
                            marker=dict(color='red', size=8, symbol='x'),
                            name='이상 탐지',
                            showlegend=True
                        )
                    )
            
            # 레이아웃 설정
            fig.update_layout(
                height=500,
                title=f'실시간 데이터 모니터링 - {feature_col} (현재 인덱스: {current_index})',
                xaxis_title='데이터 인덱스',
                yaxis_title=feature_col,
                showlegend=True,
                hovermode='x unified'
            )
            
            # x축 범위 설정
            fig.update_xaxes(range=[start_idx, end_idx])
            
            return fig
    except Exception as e:
        # 오류 발생 시 빈 차트 반환
        fig = go.Figure()
        fig.add_annotation(
            text=f"차트 생성 오류: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(height=500, title="차트 생성 오류")
        return fig

# 메인 애플리케이션
def main():
    st.markdown('<h1 class="main-header">🔍 실시간 이상 탐지 시스템 (PPO)</h1>', unsafe_allow_html=True)
    
    # 사이드바 설정
    st.sidebar.title("⚙️ 설정")

    # 설정 옵션
    auto_update = st.sidebar.checkbox("자동 업데이트", value=False)
    if auto_update:
        update_interval = st.sidebar.slider("업데이트 간격 (초)", 0.1, 2.0, 0.5, 0.1)
    window_size = st.sidebar.slider("차트 윈도우 크기", 50, 200, 100)


    # 시작/정지 버튼
    if 'running' not in st.session_state:
        st.session_state.running = False
    
    col1, col2, col3 = st.sidebar.columns(3)
    
    if col1.button("▶️ 시작"):
        st.session_state.running = True
    
    if col2.button("⏸️ 정지"):
        st.session_state.running = False
    
    if col3.button("🔄 리셋"):
        st.session_state.running = False
        # 상태 리셋
        if 'anomaly_logs' in st.session_state:
            del st.session_state.anomaly_logs
        if 'current_data_index' in st.session_state:
            del st.session_state.current_data_index


        if 'eval_env' in st.session_state:
            del st.session_state.eval_env
        if 'model' in st.session_state:
            del st.session_state.model
        if 'ui_initialized' in st.session_state:
            del st.session_state.ui_initialized

        st.rerun()
    
    # 초기화
    if 'anomaly_logs' not in st.session_state:
        st.session_state.anomaly_logs = []
    
    if 'current_data_index' not in st.session_state:
        st.session_state.current_data_index = 0
    
    # 설정값들
    test_dataset = 'sensor_data_with_anomalylabel_isolationforest'
    model_path = './log/model.pth'

    # 데이터 로드
    data_path = os.path.join('./data', test_dataset + '.csv')
    data = load_sensor_data(data_path)
    if data is None:
        st.error("데이터를 로드할 수 없습니다.")
        return
    
    # 데이터 로드 정보 표시
    st.info(f"데이터 로드 완료: {len(data)}개 (전체 데이터셋)")

    # PPO 모델 로드
    if 'model' not in st.session_state:
        try:
            st.session_state.model = PPO.load(model_path)
            st.success("PPO 모델 로드 완료")
            
            # 모델의 입력 차원 확인
            policy_input_dim = st.session_state.model.policy.input_layer.in_features
            st.info(f"PPO 모델 입력 차원: {policy_input_dim}")
            
        except Exception as e:
            st.error(f"모델 로드 실패: {e}")
            return

    # 평가 환경 생성 (evaluate.py와 동일한 방식)
    if 'eval_env' not in st.session_state:
        try:
            data_path = os.path.join('./data', test_dataset + '.csv')
            st.session_state.eval_env = make_eval_env(datapath=data_path, budget=None)  # budget=None으로 전체 데이터 탐색
            st.success("평가 환경 생성 완료")
        except Exception as e:
            st.error(f"평가 환경 생성 실패: {e}")
            return

    # 모든 UI 요소를 placeholder로 관리하여 중복 방지
    if 'ui_initialized' not in st.session_state:
        st.session_state.ui_initialized = False
    
    if not st.session_state.ui_initialized:
        # UI 초기화 (한 번만 실행)
        st.session_state.chart_placeholder = st.empty()
        st.session_state.status_placeholder = st.empty()
        st.session_state.metrics_placeholder = st.empty()
        st.session_state.ui_initialized = True
    
    # 차트 영역
    with st.session_state.chart_placeholder.container():
        st.subheader("📊 실시간 데이터 모니터링")
    
    # 상태 메트릭 영역
    with st.session_state.status_placeholder.container():
        status_col1, status_col2, status_col3, status_col4 = st.columns(4)
        
        with status_col1:
            st.metric("현재 인덱스", st.session_state.current_data_index)
        
        with status_col2:
            st.metric("총 데이터 수", len(data))
        
        with status_col3:
            if 'eval_env' in st.session_state:
                st.metric("탐지된 이상 수", len(st.session_state.eval_env.anomalies))
            else:
                st.metric("탐지된 이상 수", 0)
        
        with status_col4:
            st.metric("진행률", f"{st.session_state.current_data_index/len(data)*100:.1f}%")
    
    # 성능 지표 영역
    with st.session_state.metrics_placeholder.container():
        st.subheader("📊 성능 지표")
    
    # 이상 탐지 목록 영역
    with st.container():
        st.subheader("🚨 이상 탐지 목록")
    
    # 차트 업데이트 (고정 위치에 배치)
    try:
        # 환경에서 탐지된 이상 포인트들 가져오기
        anomaly_indices = st.session_state.eval_env.anomalies if 'eval_env' in st.session_state else []
        fig = create_realtime_chart(
            data, 
            st.session_state.current_data_index, 
            window_size, 
            anomaly_indices,
            st.session_state.running
        )
        st.session_state.chart_placeholder.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"차트 생성 중 오류: {e}")
        import traceback
        st.error(traceback.format_exc())
    
    # 성능 지표 업데이트 (항상 표시)
    total_detected = len(st.session_state.eval_env.anomalies) if 'eval_env' in st.session_state else 0
    total_processed = len(st.session_state.anomaly_logs)
    
    if total_processed > 0:
        anomaly_rate = total_detected / total_processed
    else:
        anomaly_rate = 0.0
    
    # 현재 데이터 포인트의 값
    feature_cols = [col for col in data.columns if col not in ['key', 'anomaly_label', 'anomaly_score']]
    if st.session_state.current_data_index < len(data):
        if 'Pressure' in feature_cols:
            current_value = data.iloc[st.session_state.current_data_index]['Pressure']
        else:
            current_value = data.iloc[st.session_state.current_data_index][feature_cols[0]] if feature_cols else st.session_state.current_data_index
    else:
        current_value = 0.0
    
    # 성능 지표 계산 (evaluate_policy와 정확히 동일한 방식)
    tp = sum(1 for log in st.session_state.anomaly_logs if log.get('is_anomaly', False) and log.get('ground_truth', 0) == 0)  # 0: anomaly
    fp = sum(1 for log in st.session_state.anomaly_logs if log.get('is_anomaly', False) and log.get('ground_truth', 0) == 1)  # 1: normal
    tn = sum(1 for log in st.session_state.anomaly_logs if not log.get('is_anomaly', False) and log.get('ground_truth', 0) == 1)  # 1: normal
    fn = sum(1 for log in st.session_state.anomaly_logs if not log.get('is_anomaly', False) and log.get('ground_truth', 0) == 0)  # 0: anomaly
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    if st.session_state.running:
        st.session_state.metrics_placeholder.markdown(f"""
        <div class="performance-metrics">
            <strong>📈 실시간 성능 지표</strong><br>
            총 처리 수: {total_processed}<br>
            탐지된 이상: {total_detected}<br>
            이상 탐지율: {anomaly_rate:.4f}<br>
            현재 값: {current_value:.4f}<br><br>
            <strong>📊 상세 성능 지표</strong><br>
            정밀도 (Precision): {precision:.4f}<br>
            재현율 (Recall): {recall:.4f}<br>
            F1 점수: {f1_score:.4f}<br>
            TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}
        </div>
        """, unsafe_allow_html=True)
    else:
        # 시작 전에는 기본 정보만 표시
        st.session_state.metrics_placeholder.info("🚀 시작 버튼을 눌러 실시간 이상 탐지를 시작하세요!")
    
    # 이상 탐지 목록 표시 (고정 위치)
    anomaly_logs = [log for log in st.session_state.anomaly_logs if log.get('is_anomaly', False)]
    if anomaly_logs:
        # 이상 탐지된 항목들만 테이블로 표시
        anomaly_table_data = []
        for log in anomaly_logs:
            ground_truth_info = "이상" if log.get('ground_truth', 0) == 0 else "정상"
            anomaly_table_data.append({
                '인덱스': log['index'],
                '탐지 시각': log['timestamp'],
                '센서 값': f"{log['value']:.4f}",
                '실제': ground_truth_info
            })
        
        # DataFrame으로 변환하여 테이블 표시
        df_anomaly = pd.DataFrame(anomaly_table_data)
        st.dataframe(df_anomaly, use_container_width=True)
        
        # 요약 정보
        st.info(f"총 {len(anomaly_logs)}개의 이상이 탐지되었습니다.")
    else:
        st.info("아직 이상이 탐지되지 않았습니다.")
    
    # 실시간 데이터 처리 및 PPO 모델 예측
    if st.session_state.running:
        try:
            # PPO 모델로 순차적 예측
            is_anomaly, anomaly_prob, normal_prob, reward, ground_truth, selected_action, selected_value, done = predict_anomaly_sequential(
                st.session_state.model, 
                st.session_state.eval_env,
                data,
                st.session_state.current_data_index
            )
            
            # 예측 결과 처리
            log_entry = {
                'index': selected_action,
                'timestamp': time.strftime('%H:%M:%S'),
                'value': selected_value,
                'anomaly_prob': anomaly_prob,
                'normal_prob': normal_prob,
                'is_anomaly': is_anomaly,
                'reward': reward,
                'ground_truth': ground_truth
            }
            st.session_state.anomaly_logs.append(log_entry)
            
            # 현재 인덱스 증가 (순차적 처리)
            st.session_state.current_data_index += 1
            
            # 모든 데이터 처리 완료 시 종료
            if st.session_state.current_data_index >= len(data):
                st.session_state.running = False
                st.success("🎉 모든 데이터 처리가 완료되었습니다!")
                
        except Exception as e:
            st.error(f"PPO 모델 예측 중 오류: {e}")
            import traceback
            st.error(traceback.format_exc())
        
        # 업데이트 방식 선택
        if auto_update:
            # 자동 업데이트
            time.sleep(update_interval)
            st.rerun()
        else:
            # 수동 업데이트
            if st.button("🔄 다음 단계 처리", key="next_step"):
                pass  # 버튼을 누르면 다음 단계로 진행
            else:
                st.info("🔄 버튼을 눌러 다음 데이터를 처리하세요.")
                st.stop()  # 버튼을 누르지 않으면 여기서 멈춤

if __name__ == "__main__":
    main()