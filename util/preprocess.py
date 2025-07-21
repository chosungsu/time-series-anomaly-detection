import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from .eda import (
    TimeSeriesIsolationForestDetector
)

def postprocess_isolation_labels(labels, window_size, min_anomaly_count=3):
    """
    window_size 내에 이상치가 min_anomaly_count개 이상이면
    가장 작은 인덱스만 이상(1)으로, 나머지는 정상(0)으로 처리
    이후 탐색 구간은 마지막 이상치 인덱스 다음부터 window_size만큼 다시 탐색
    """
    labels = np.array(labels).copy()
    n = len(labels)
    processed = np.zeros(n, dtype=int)
    idx = 0
    while idx <= n - window_size:
        window = labels[idx:idx+window_size]
        anomaly_indices = np.where(window == 1)[0]
        if len(anomaly_indices) >= min_anomaly_count:
            # 가장 작은 인덱스만 이상(1), 나머지는 정상(0)
            first_anom = anomaly_indices[0]
            processed[idx + first_anom] = 1
            # 나머지는 0으로(이미 0이므로 따로 처리 불필요)
            # 다음 탐색 구간은 마지막 이상치 인덱스 다음부터
            idx = idx + anomaly_indices[-1] + 1
        else:
            idx += 1
    # 나머지 구간(윈도우 크기 미만)은 모두 정상(0)
    return processed

def save_anomaly_labels(df, model_name, window_size=7, save_dir=None):
    """
    model.py의 이상탐지 모델로 정상/이상 라벨을 예측하여 data 폴더에 csv로 저장
    model_name:'isolationforest'
    """
    feature_cols = [col for col in df.columns if col != 'key']
    if model_name == 'isolationforest':
        detector = TimeSeriesIsolationForestDetector(
            window_size=window_size,
            params={
                'n_estimators': 200,
                'contamination': 0.02,
                'max_samples': 'auto',  # 데이터 개수보다 크지 않게
                'max_features': 1.0,
                'random_state': 42
            }
        )
        labels, scores = detector.fit_predict(df, feature_cols=feature_cols, key_col='key')
        # 라벨 변환: 항상 0(정상), 1(이상)으로 맞춤
        if np.any(labels == -1):
            anomaly_label = np.where(labels == -1, 1, 0)
        else:
            anomaly_label = labels.copy()
        # post-process: window 내 3개 이상 이상치면 첫번째만 1, 나머지 0
        anomaly_label = postprocess_isolation_labels(anomaly_label, window_size, min_anomaly_count=3)
    else:
        raise ValueError(f"지원하지 않는 모델: {model_name}")
    result_df = df.copy()
    result_df['anomaly_label'] = anomaly_label
    result_df['anomaly_score'] = scores
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'sensor_data_with_anomalylabel_{model_name}.csv')
    result_df.to_csv(save_path, index=False)
    print(f"[{model_name}] 이상탐지 라벨 저장 완료: {save_path}")
