import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
import random
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def set_seed(seed=42):
    """랜덤 시드를 설정합니다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class DeepSVDDNet(nn.Module):
    """
    Deep SVDD를 위한 신경망 구조
    
    시계열 데이터의 특징을 추출하고 압축하는 인코더 네트워크입니다.
    """
    
    def __init__(self, input_dim: int, hidden_dims: list = None, latent_dim: int = 32):
        """
        Deep SVDD 네트워크 초기화
        
        Args:
            input_dim (int): 입력 특징의 차원
            hidden_dims (list): 은닉층의 차원들
            latent_dim (int): 잠재 공간의 차원
        """
        super(DeepSVDDNet, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 64]
        
        # 인코더 레이어들
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # 최종 잠재 공간으로의 매핑
        layers.append(nn.Linear(prev_dim, latent_dim))
        
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파
        
        Args:
            x (torch.Tensor): 입력 데이터
            
        Returns:
            torch.Tensor: 잠재 공간 표현
        """
        return self.encoder(x)


class DeepSVDDAnomalyDetector:
    """
    Deep SVDD (Deep Support Vector Data Description) 이상탐지 모델
    
    딥러닝을 사용하여 정상 데이터의 특징 공간을 학습하고,
    정상 영역 밖의 데이터를 이상치로 탐지합니다.
    """
    
    def __init__(self, 
                 input_dim: int = None,
                 hidden_dims: list = None,
                 latent_dim: int = 32,
                 nu: float = 0.1,
                 window_size: int = 10,
                 feature_method: str = 'statistical',
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 epochs: int = 100,
                 device: str = 'auto'):
        """
        Deep SVDD 이상탐지 모델 초기화
        
        Args:
            input_dim (int): 입력 특징의 차원 (None이면 자동 설정)
            hidden_dims (list): 은닉층의 차원들
            latent_dim (int): 잠재 공간의 차원
            nu (float): 이상치로 예상되는 데이터의 비율 (0.0 ~ 1.0)
            window_size (int): 시계열 윈도우 크기
            feature_method (str): 특징 추출 방법 ('statistical', 'temporal', 'combined')
            learning_rate (float): 학습률
            batch_size (int): 배치 크기
            epochs (int): 학습 에포크 수
            device (str): 사용할 디바이스 ('auto', 'cpu', 'cuda')
        """
        # 모델 초기화 전에 seed 설정
        set_seed(200)
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.nu = nu
        self.window_size = window_size
        self.feature_method = feature_method
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        # 디바이스 설정
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
        self.center = None
        
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
            
            # 모든 특징을 결합
            feature_vector = np.concatenate([
                [mean_val, std_val, min_val, max_val, median_val, q25, q75, iqr, skewness, kurtosis],
                diff_from_mean
            ])
            
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
    
    def _train_model(self, features: np.ndarray):
        """
        Deep SVDD 모델을 학습합니다.
        
        Args:
            features (np.ndarray): 학습할 특징들
        """
        # 입력 차원 설정
        if self.input_dim is None:
            self.input_dim = features.shape[1]
        
        # 모델 초기화
        self.model = DeepSVDDNet(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            latent_dim=self.latent_dim
        ).to(self.device)
        
        # 데이터를 텐서로 변환
        features_tensor = torch.FloatTensor(features).to(self.device)
        dataset = TensorDataset(features_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 옵티마이저 설정
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # 학습 루프
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            
            for batch_features in dataloader:
                batch_features = batch_features[0]
                
                # 순전파
                encoded = self.model(batch_features)
                
                # Deep SVDD 손실: 정상 데이터를 중심으로 하는 구의 반지름을 최소화
                if self.center is None:
                    # 첫 번째 배치에서 중심 초기화
                    self.center = encoded.mean(dim=0).detach()
                
                # 중심으로부터의 거리
                distances = torch.sum((encoded - self.center) ** 2, dim=1)
                
                # 손실 계산 (Soft-boundary SVDD)
                radius = torch.quantile(distances, 1 - self.nu)
                loss = torch.mean(torch.max(torch.zeros_like(distances), distances - radius))
                
                # 역전파
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # 진행 상황 출력
            if (epoch + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {total_loss/len(dataloader):.6f}")
        
        # 최종 중심 계산
        self.model.eval()
        with torch.no_grad():
            all_encoded = self.model(features_tensor)
            self.center = all_encoded.mean(dim=0)
    
    def fit(self, data: pd.DataFrame) -> 'DeepSVDDAnomalyDetector':
        """
        Deep SVDD 모델을 학습합니다.
        
        Args:
            data (pd.DataFrame): 학습할 시계열 데이터
            
        Returns:
            DeepSVDDAnomalyDetector: 학습된 모델 인스턴스
        """
        if data.empty:
            raise ValueError("데이터가 비어있습니다.")
        
        self.feature_names = data.columns.tolist()
        
        # 특징 추출
        features = self._create_features(data)
        
        # 특징 정규화
        features_scaled = self.scaler.fit_transform(features)
        
        # 모델 학습
        print("Deep SVDD 모델 학습 중...")
        self._train_model(features_scaled)
        
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
        self.model.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features_scaled).to(self.device)
            encoded = self.model(features_tensor)
            
            # 중심으로부터의 거리
            distances = torch.sum((encoded - self.center) ** 2, dim=1)
            
            # 임계값 설정 (nu 비율에 따라)
            threshold = torch.quantile(distances, 1 - self.nu)
            
            # 이상치 예측
            predictions = torch.where(distances > threshold, -1, 1).cpu().numpy()
        
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
        
        # 점수 계산
        self.model.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features_scaled).to(self.device)
            encoded = self.model(features_tensor)
            
            # 중심으로부터의 거리
            distances = torch.sum((encoded - self.center) ** 2, dim=1)
            
            # 점수를 양수로 변환 (높을수록 이상)
            scores = distances.cpu().numpy()
        
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
        axes[1].set_title('Deep SVDD 이상 점수')
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
        
        axes[2].set_title('Deep SVDD 이상치 탐지 결과')
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
            '잠재 공간 차원': self.latent_dim,
            'nu 파라미터': self.nu
        }
        
        return summary
    
    def get_center(self) -> np.ndarray:
        """
        학습된 모델의 정상 데이터 중심을 반환합니다.
        
        Returns:
            np.ndarray: 정상 데이터 중심
        """
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다.")
        
        return self.center.cpu().numpy()
