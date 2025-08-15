import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
import random


def set_seed(seed=42):
    """랜덤 시드를 설정합니다."""
    random.seed(seed)
    np.random.seed(seed)


class LOFAnomalyDetector:
    """
    Local Outlier Factor (LOF)를 사용한 시계열 이상탐지 모델
    
    LOF는 데이터 포인트의 밀도를 기반으로 이상치를 탐지하는 비지도 학습 방법입니다.
    시계열 데이터의 각 시점에서 주변 이웃들과의 밀도 차이를 계산하여 이상치를 식별합니다.
    """
    
    def __init__(self, 
                 n_neighbors: int = 20,
                 contamination: float = 0.1,
                 metric: str = 'euclidean',
                 window_size: int = 10,
                 threshold: Optional[float] = None):
        """
        LOF 이상탐지 모델 초기화
        
        Args:
            n_neighbors (int): LOF 계산에 사용할 이웃의 수
            contamination (float): 이상치로 예상되는 데이터의 비율 (0.0 ~ 1.0)
            metric (str): 거리 측정 방법 ('euclidean', 'manhattan', 'cosine' 등)
            window_size (int): 시계열 윈도우 크기 (이동 평균 계산용)
            threshold (float, optional): 이상치 판별 임계값 (None이면 자동 설정)
        """
        # 모델 초기화 전에 seed 설정
        set_seed(200)
        
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.metric = metric
        self.window_size = window_size
        self.threshold = threshold
        
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
        
    def _create_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        시계열 데이터에서 특징을 추출합니다.
        
        Args:
            data (pd.DataFrame): 원본 시계열 데이터
            
        Returns:
            np.ndarray: 추출된 특징들
        """
        features = []
        
        for i in range(len(data)):
            if i < self.window_size - 1:
                # 윈도우가 충분하지 않은 경우 0으로 채움
                window_data = np.zeros(self.window_size * len(data.columns))
                window_data[-(i+1)*len(data.columns):] = data.iloc[:i+1].values.flatten()
            else:
                # 윈도우 크기만큼의 데이터 사용
                window_data = data.iloc[i-self.window_size+1:i+1].values.flatten()
            
            # 통계적 특징 추출
            mean_val = np.mean(window_data)
            std_val = np.std(window_data)
            min_val = np.min(window_data)
            max_val = np.max(window_data)
            
            # 현재 값과 윈도우 평균의 차이
            current_val = data.iloc[i].values
            diff_from_mean = current_val - mean_val
            
            # 변화율 (이전 값과의 차이)
            if i > 0:
                change_rate = current_val - data.iloc[i-1].values
            else:
                change_rate = np.zeros_like(current_val)
            
            # 모든 특징을 결합
            feature_vector = np.concatenate([
                [mean_val, std_val, min_val, max_val],
                diff_from_mean,
                change_rate
            ])
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def fit(self, data: pd.DataFrame) -> 'LOFAnomalyDetector':
        """
        LOF 모델을 학습합니다.
        
        Args:
            data (pd.DataFrame): 학습할 시계열 데이터
            
        Returns:
            LOFAnomalyDetector: 학습된 모델 인스턴스
        """
        if data.empty:
            raise ValueError("데이터가 비어있습니다.")
        
        self.feature_names = data.columns.tolist()
        
        # 특징 추출
        features = self._create_features(data)
        
        # 특징 정규화
        features_scaled = self.scaler.fit_transform(features)
        
        # LOF 모델 학습
        self.model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            metric=self.metric,
            novelty=False  # fit_predict 사용을 위해 False
        )
        
        # 모델 학습 (LOF는 fit_predict를 사용하여 학습과 예측을 동시에 수행)
        self.model.fit_predict(features_scaled)
        
        self.is_fitted = True
        return self
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        학습된 모델을 사용하여 이상치를 예측합니다.
        
        Args:
            data (pd.DataFrame): 예측할 시계열 데이터
            
        Returns:
            np.ndarray: 이상치 예측 결과 (-1: 이상치, 1: 정상)
        """
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다. fit() 메서드를 먼저 호출하세요.")
        
        if data.empty:
            raise ValueError("데이터가 비어있습니다.")
        
        # 특징 추출
        features = self._create_features(data)
        
        # 특징 정규화
        features_scaled = self.scaler.transform(features)
        
        # 예측 수행
        predictions = self.model.fit_predict(features_scaled)
        
        return predictions
    
    def score_samples(self, data: pd.DataFrame) -> np.ndarray:
        """
        각 데이터 포인트의 이상 점수를 계산합니다.
        
        Args:
            data (pd.DataFrame): 점수를 계산할 시계열 데이터
            
        Returns:
            np.ndarray: 각 데이터 포인트의 이상 점수 (높을수록 이상)
        """
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다. fit() 메서드를 먼저 호출하세요.")
        
        if data.empty:
            raise ValueError("데이터가 비어있습니다.")
        
        # 특징 추출
        features = self._create_features(data)
        
        # 특징 정규화
        features_scaled = self.scaler.transform(features)
        
        # LOF 점수 계산 (음수 값이므로 절댓값을 취함)
        scores = np.abs(self.model.negative_outlier_factor_)
        
        return scores
    
    def detect_anomalies(self, data: pd.DataFrame, 
                        threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        이상치를 탐지하고 점수를 반환합니다.
        
        Args:
            data (pd.DataFrame): 탐지할 시계열 데이터
            threshold (float, optional): 이상치 판별 임계값
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (이상치 예측, 이상 점수)
        """
        predictions = self.predict(data)
        scores = self.score_samples(data)
        
        # 임계값이 설정되지 않은 경우 자동으로 설정
        if threshold is None:
            if self.threshold is None:
                # 점수의 95 퍼센타일을 임계값으로 사용
                threshold = np.percentile(scores, 95)
            else:
                threshold = self.threshold
        
        # 임계값 기반으로 이상치 재분류
        anomaly_predictions = (scores > threshold).astype(int)
        anomaly_predictions[anomaly_predictions == 1] = -1  # 이상치를 -1로 표시
        anomaly_predictions[anomaly_predictions == 0] = 1   # 정상을 1로 표시
        
        return anomaly_predictions, scores
    
    def plot_results(self, data: pd.DataFrame, 
                    predictions: np.ndarray, 
                    scores: np.ndarray,
                    save_path: Optional[str] = None) -> None:
        """
        이상탐지 결과를 시각화합니다.
        
        Args:
            data (pd.DataFrame): 원본 데이터
            predictions (np.ndarray): 이상치 예측 결과
            scores (np.ndarray): 이상 점수
            save_path (str, optional): 저장할 파일 경로
        """
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # 원본 데이터 플롯
        for col in data.columns:
            axes[0].plot(data.index, data[col], label=col, alpha=0.7)
        axes[0].set_title('원본 시계열 데이터')
        axes[0].set_xlabel('시간')
        axes[0].set_ylabel('값')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 이상 점수 플롯
        axes[1].plot(data.index, scores, 'b-', alpha=0.7, label='이상 점수')
        if self.threshold:
            axes[1].axhline(y=self.threshold, color='r', linestyle='--', 
                           label=f'임계값: {self.threshold:.3f}')
        axes[1].set_title('이상 점수')
        axes[1].set_xlabel('시간')
        axes[1].set_ylabel('이상 점수')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 이상치 탐지 결과 플롯
        anomaly_indices = np.where(predictions == -1)[0]
        normal_indices = np.where(predictions == 1)[0]
        
        # 정상 데이터 플롯
        if len(normal_indices) > 0:
            for col in data.columns:
                axes[2].scatter(normal_indices, data.iloc[normal_indices][col], 
                              c='blue', alpha=0.6, s=20, label=f'{col} (정상)' if col == data.columns[0] else "")
        
        # 이상치 데이터 플롯
        if len(anomaly_indices) > 0:
            for col in data.columns:
                axes[2].scatter(anomaly_indices, data.iloc[anomaly_indices][col], 
                              c='red', alpha=0.8, s=50, marker='x', 
                              label=f'{col} (이상)' if col == data.columns[0] else "")
        
        axes[2].set_title('이상치 탐지 결과')
        axes[2].set_xlabel('시간')
        axes[2].set_ylabel('값')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    def get_anomaly_summary(self, data: pd.DataFrame, 
                           predictions: np.ndarray, 
                           scores: np.ndarray) -> dict:
        """
        이상탐지 결과의 요약 통계를 반환합니다.
        
        Args:
            data (pd.DataFrame): 원본 데이터
            predictions (np.ndarray): 이상치 예측 결과
            scores (np.ndarray): 이상 점수
            
        Returns:
            dict: 요약 통계 정보
        """
        total_points = len(data)
        anomaly_points = np.sum(predictions == -1)
        normal_points = np.sum(predictions == 1)
        
        anomaly_ratio = anomaly_points / total_points
        
        summary = {
            '총 데이터 포인트': total_points,
            '정상 데이터 포인트': normal_points,
            '이상 데이터 포인트': anomaly_points,
            '이상 비율': f"{anomaly_ratio:.2%}",
            '최대 이상 점수': np.max(scores),
            '최소 이상 점수': np.min(scores),
            '평균 이상 점수': np.mean(scores),
            '이상 점수 표준편차': np.std(scores)
        }
        
        return summary
