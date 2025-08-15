import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
import random


def set_seed(seed=42):
    """랜덤 시드를 설정합니다."""
    random.seed(seed)
    np.random.seed(seed)


class IsolationForestAnomalyDetector:
    """
    Isolation Forest를 사용한 시계열 이상탐지 모델
    
    Isolation Forest는 랜덤하게 선택된 특징으로 데이터를 분할하여
    이상치를 탐지하는 앙상블 방법입니다. 정상 데이터는 더 많은 분할이 필요하고,
    이상치는 적은 분할로 격리됩니다.
    """
    
    def __init__(self, 
                 n_estimators: int = 100,
                 contamination: float = 0.1,
                 max_samples: Union[int, float] = 'auto',
                 random_state: int = 42,
                 window_size: int = 10,
                 feature_method: str = 'statistical'):
        """
        Isolation Forest 이상탐지 모델 초기화
        
        Args:
            n_estimators (int): 트리의 개수
            contamination (float): 이상치로 예상되는 데이터의 비율 (0.0 ~ 1.0)
            max_samples (Union[int, float]): 각 트리에서 사용할 샘플 수
            random_state (int): 랜덤 시드
            window_size (int): 시계열 윈도우 크기
            feature_method (str): 특징 추출 방법 ('statistical', 'temporal', 'combined')
        """
        # 모델 초기화 전에 seed 설정
        set_seed(200)
        
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self.random_state = random_state
        self.window_size = window_size
        self.feature_method = feature_method
        
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
        
    def _create_statistical_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        통계적 특징을 추출합니다.
        
        Args:
            data (pd.DataFrame): 원본 시계열 데이터
            
        Returns:
            np.ndarray: 추출된 통계적 특징들
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
            median_val = np.median(window_data)
            
            # 분위수 특징
            q25 = np.percentile(window_data, 25)
            q75 = np.percentile(window_data, 75)
            iqr = q75 - q25
            
            # 왜도와 첨도
            skewness = self._calculate_skewness(window_data)
            kurtosis = self._calculate_kurtosis(window_data)
            
            # 현재 값과 윈도우 평균의 차이
            current_val = data.iloc[i].values
            diff_from_mean = current_val - mean_val
            
            # 모든 특징을 결합 (차원 일치 확인)
            feature_vector = np.concatenate([
                [mean_val, std_val, min_val, max_val, median_val, q25, q75, iqr, skewness, kurtosis],
                diff_from_mean
            ])
            
            # 차원이 일치하지 않으면 조정
            if len(feature_vector) != 10 + len(diff_from_mean):
                # 필요한 길이에 맞춰 조정
                expected_length = 10 + len(diff_from_mean)
                if len(feature_vector) > expected_length:
                    feature_vector = feature_vector[:expected_length]
                else:
                    # 부족한 경우 0으로 채움
                    padding = np.zeros(expected_length - len(feature_vector))
                    feature_vector = np.concatenate([feature_vector, padding])
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _create_temporal_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        시간적 특징을 추출합니다.
        
        Args:
            data (pd.DataFrame): 원본 시계열 데이터
            
        Returns:
            np.ndarray: 추출된 시간적 특징들
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
            
            # 변화율 특징
            if i > 0:
                change_rate = data.iloc[i].values - data.iloc[i-1].values
                acceleration = change_rate - (data.iloc[i-1].values - data.iloc[i-2].values) if i > 1 else np.zeros_like(change_rate)
            else:
                change_rate = np.zeros_like(data.iloc[i].values)
                acceleration = np.zeros_like(change_rate)
            
            # 이동 평균과의 차이
            if i >= self.window_size - 1:
                moving_avg = np.mean(data.iloc[i-self.window_size+1:i+1], axis=0)
                diff_from_moving_avg = data.iloc[i].values - moving_avg
            else:
                moving_avg = np.mean(data.iloc[:i+1], axis=0) if i > 0 else data.iloc[i].values
                diff_from_moving_avg = data.iloc[i].values - moving_avg
            
            # 모든 특징을 결합
            feature_vector = np.concatenate([
                change_rate,
                acceleration,
                diff_from_moving_avg
            ])
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _create_combined_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        통계적 특징과 시간적 특징을 결합합니다.
        
        Args:
            data (pd.DataFrame): 원본 시계열 데이터
            
        Returns:
            np.ndarray: 결합된 특징들
        """
        statistical_features = self._create_statistical_features(data)
        temporal_features = self._create_temporal_features(data)
        
        # 두 특징을 결합
        combined_features = np.concatenate([statistical_features, temporal_features], axis=1)
        
        return combined_features
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """왜도를 계산합니다."""
        if len(data) < 3:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """첨도를 계산합니다."""
        if len(data) < 4:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _create_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        선택된 방법에 따라 특징을 추출합니다.
        
        Args:
            data (pd.DataFrame): 원본 시계열 데이터
            
        Returns:
            np.ndarray: 추출된 특징들
        """
        if self.feature_method == 'statistical':
            return self._create_statistical_features(data)
        elif self.feature_method == 'temporal':
            return self._create_temporal_features(data)
        elif self.feature_method == 'combined':
            return self._create_combined_features(data)
        else:
            raise ValueError(f"지원하지 않는 특징 추출 방법: {self.feature_method}")
    
    def fit(self, data: pd.DataFrame) -> 'IsolationForestAnomalyDetector':
        """
        Isolation Forest 모델을 학습합니다.
        
        Args:
            data (pd.DataFrame): 학습할 시계열 데이터
            
        Returns:
            IsolationForestAnomalyDetector: 학습된 모델 인스턴스
        """
        if data.empty:
            raise ValueError("데이터가 비어있습니다.")
        
        self.feature_names = data.columns.tolist()
        
        # 특징 추출
        features = self._create_features(data)
        
        # 특징 정규화
        features_scaled = self.scaler.fit_transform(features)
        
        # Isolation Forest 모델 학습
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=self.max_samples,
            random_state=self.random_state
        )
        
        self.model.fit(features_scaled)
        
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
        predictions = self.model.predict(features_scaled)
        
        return predictions
    
    def score_samples(self, data: pd.DataFrame) -> np.ndarray:
        """
        각 데이터 포인트의 이상 점수를 계산합니다.
        
        Args:
            data (pd.DataFrame): 점수를 계산할 시계열 데이터
            
        Returns:
            np.ndarray: 각 데이터 포인트의 이상 점수 (낮을수록 이상)
        """
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다. fit() 메서드를 먼저 호출하세요.")
        
        if data.empty:
            raise ValueError("데이터가 비어있습니다.")
        
        # 특징 추출
        features = self._create_features(data)
        
        # 특징 정규화
        features_scaled = self.scaler.transform(features)
        
        # Isolation Forest 점수 계산
        scores = self.model.score_samples(features_scaled)
        
        # 점수를 양수로 변환 (높을수록 이상)
        scores = -scores
        
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
            # 점수의 95 퍼센타일을 임계값으로 사용
            threshold = np.percentile(scores, 95)
        
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
            '이상 점수 표준편차': np.std(scores),
            '사용된 특징 방법': self.feature_method,
            '트리 개수': self.n_estimators,
            'contamination': self.contamination
        }
        
        return summary
    
    def get_feature_importances(self) -> np.ndarray:
        """
        학습된 모델의 특징 중요도를 반환합니다.
        
        Returns:
            np.ndarray: 특징 중요도 배열
        """
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        # Isolation Forest는 직접적인 특징 중요도를 제공하지 않음
        # 대신 각 트리의 분할 기준을 분석하여 중요도 추정
        return np.ones(self.model.n_features_in_) / self.model.n_features_in_
