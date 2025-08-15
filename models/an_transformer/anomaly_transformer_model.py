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
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

from .encoder import Encoder, EncoderLayer
from .attn import AnomalyAttention, AttentionLayer
from .embed import DataEmbedding


class AnomalyTransformer(nn.Module):
    def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, activation='gelu', output_attention=True):
        super(AnomalyTransformer, self).__init__()
        self.output_attention = output_attention

        # Encoding
        self.embedding = DataEmbedding(enc_in, d_model, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(win_size, False, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x):
        enc_out = self.embedding(x)
        enc_out, series, prior, sigmas = self.encoder(enc_out)
        enc_out = self.projection(enc_out)

        if self.output_attention:
            return enc_out, series, prior, sigmas
        else:
            return enc_out  # [B, L, D]


class AnomalyTransformerDetector:
    """
    Anomaly Transformer 이상탐지 모델
    
    Transformer 아키텍처를 사용하여 시계열 데이터의 정상 패턴을 학습하고,
    어텐션 가중치와 재구성 오차를 기반으로 이상치를 탐지합니다.
    """
    
    def __init__(self, 
                 input_dim: int = None,
                 d_model: int = 128,
                 n_heads: int = 8,
                 e_layers: int = 3,
                 d_ff: int = 512,
                 window_size: int = 10,
                 feature_method: str = 'statistical',
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 epochs: int = 50,
                 device: str = 'auto',
                 dropout: float = 0.1,
                 activation: str = 'gelu'):
        """
        Anomaly Transformer 이상탐지 모델 초기화
        
        Args:
            input_dim (int): 입력 특징의 차원 (None이면 자동 설정)
            d_model (int): 모델의 차원
            n_heads (int): 멀티헤드 어텐션의 헤드 수
            e_layers (int): Transformer 레이어 수
            d_ff (int): 피드포워드 네트워크의 차원
            window_size (int): 시계열 윈도우 크기
            feature_method (str): 특징 추출 방법 ('statistical', 'temporal', 'combined')
            learning_rate (float): 학습률
            batch_size (int): 배치 크기
            epochs (int): 학습 에포크 수
            device (str): 사용할 디바이스 ('auto', 'cpu', 'cuda')
            dropout (float): 드롭아웃 비율
            activation (str): 활성화 함수 ('relu', 'gelu')
        """
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_ff = d_ff
        self.window_size = window_size
        self.feature_method = feature_method
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout = dropout
        self.activation = activation
        
        # 디바이스 설정
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
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
            
            # 통계적 특징 추출 (NaN 값 안전 처리)
            mean_val = np.nanmean(window_data)
            std_val = np.nanstd(window_data)
            min_val = np.nanmin(window_data)
            max_val = np.nanmax(window_data)
            median_val = np.nanmedian(window_data)
            
            # 분위수 특징 (NaN 값 안전 처리)
            q25 = np.nanpercentile(window_data, 25)
            q75 = np.nanpercentile(window_data, 75)
            iqr = q75 - q25
            
            # 왜도와 첨도
            skewness = self._calculate_skewness(window_data)
            kurtosis = self._calculate_kurtosis(window_data)
            
            # 현재 값과 윈도우 평균의 차이
            current_val = data.iloc[i].values
            diff_from_mean = current_val - mean_val
            
            # NaN 값을 0으로 대체
            if np.isnan(mean_val): mean_val = 0.0
            if np.isnan(std_val): std_val = 0.0
            if np.isnan(min_val): min_val = 0.0
            if np.isnan(max_val): max_val = 0.0
            if np.isnan(median_val): median_val = 0.0
            if np.isnan(q25): q25 = 0.0
            if np.isnan(q75): q75 = 0.0
            if np.isnan(iqr): iqr = 0.0
            if np.isnan(skewness): skewness = 0.0
            if np.isnan(kurtosis): kurtosis = 0.0
            
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
        # NaN 값 제거
        clean_data = data[~np.isnan(data)]
        if len(clean_data) < 3:
            return 0.0
        mean = np.nanmean(clean_data)
        std = np.nanstd(clean_data)
        if std == 0 or np.isnan(std):
            return 0.0
        skewness = np.nanmean(((clean_data - mean) / std) ** 3)
        return 0.0 if np.isnan(skewness) else skewness
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """첨도를 계산합니다."""
        if len(data) < 4:
            return 0.0
        # NaN 값 제거
        clean_data = data[~np.isnan(data)]
        if len(clean_data) < 4:
            return 0.0
        mean = np.nanmean(clean_data)
        std = np.nanstd(clean_data)
        if std == 0 or np.isnan(std):
            return 0.0
        kurtosis = np.nanmean(((clean_data - mean) / std) ** 4) - 3
        return 0.0 if np.isnan(kurtosis) else kurtosis
    
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
    
    def _create_sequences(self, features: np.ndarray) -> np.ndarray:
        """
        특징을 시퀀스로 변환합니다.
        
        Args:
            features (np.ndarray): 특징 배열
            
        Returns:
            np.ndarray: 시퀀스 형태의 특징들
        """
        sequences = []
        
        # 윈도우 크기가 데이터보다 큰 경우 처리
        if self.window_size > len(features):
            # 데이터를 윈도우 크기에 맞게 패딩
            padded_features = np.zeros((self.window_size, features.shape[1]))
            padded_features[:len(features)] = features
            sequences.append(padded_features)
        else:
            for i in range(len(features) - self.window_size + 1):
                sequence = features[i:i + self.window_size]
                sequences.append(sequence)
        
        return np.array(sequences)
    
    def _train_model(self, features: np.ndarray):
        """
        Anomaly Transformer 모델을 학습합니다.
        
        Args:
            features (np.ndarray): 학습할 특징들
        """
        # 입력 차원 설정
        if self.input_dim is None:
            self.input_dim = features.shape[1]
        
        # 시퀀스 생성
        sequences = self._create_sequences(features)
        
        # 모델 초기화
        self.model = AnomalyTransformer(
            win_size=self.window_size,
            enc_in=self.input_dim,
            c_out=self.input_dim,
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            activation=self.activation,
            output_attention=True
        ).to(self.device)
        
        # 데이터를 텐서로 변환
        sequences_tensor = torch.FloatTensor(sequences).to(self.device)
        dataset = TensorDataset(sequences_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 옵티마이저 설정
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # 손실 함수
        mse_loss = nn.MSELoss()
        
        # 학습 루프
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            
            for batch_sequences in dataloader:
                batch_sequences = batch_sequences[0]
                
                # 순전파
                reconstructed, series, prior, sigmas = self.model(batch_sequences)
                
                # 재구성 오차
                reconstruction_loss = mse_loss(reconstructed, batch_sequences)
                
                # Prior Loss는 복잡하므로 간단하게 처리
                # 재구성 오차만 사용
                total_batch_loss = reconstruction_loss
                
                # 역전파
                optimizer.zero_grad()
                total_batch_loss.backward()
                optimizer.step()
                
                total_loss += total_batch_loss.item()
            
            # 진행 상황 출력
            if (epoch + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {total_loss/len(dataloader):.6f}")
    
    def fit(self, data: pd.DataFrame) -> 'AnomalyTransformerDetector':
        """
        Anomaly Transformer 모델을 학습합니다.
        
        Args:
            data (pd.DataFrame): 학습할 시계열 데이터
            
        Returns:
            AnomalyTransformerDetector: 학습된 모델 인스턴스
        """
        if data.empty:
            raise ValueError("데이터가 비어있습니다.")
        
        self.feature_names = data.columns.tolist()
        
        # 특징 추출
        features = self._create_features(data)
        
        # 특징 정규화
        features_scaled = self.scaler.fit_transform(features)
        
        # 모델 학습
        print("Anomaly Transformer 모델 학습 중...")
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
        
        # 시퀀스 생성
        sequences = self._create_sequences(features_scaled)
        
        # 시퀀스가 비어있는 경우 처리
        if len(sequences) == 0:
            # 모든 점수를 0으로 설정
            scores = np.zeros(len(features))
            threshold = 0.0
            predictions = np.ones(len(features))  # 모든 데이터를 정상으로 분류
            return predictions
        
        # 예측 수행
        self.model.eval()
        with torch.no_grad():
            sequences_tensor = torch.FloatTensor(sequences).to(self.device)
            reconstructed, series, prior, sigmas = self.model(sequences_tensor)
            
            # 재구성 오차
            reconstruction_errors = torch.mean((reconstructed - sequences_tensor) ** 2, dim=-1)
            
            # Prior Loss는 복잡하므로 간단하게 처리
            # 재구성 오차만 사용하여 이상 점수 계산
            anomaly_scores = reconstruction_errors
            
            # 시퀀스 길이에 맞춰 점수 확장
            expanded_scores = []
            for i in range(len(features)):
                if i < self.window_size - 1:
                    # 윈도우가 충분하지 않은 경우 첫 번째 점수 사용
                    # anomaly_scores[0]가 텐서인 경우 평균값 사용
                    if torch.is_tensor(anomaly_scores[0]):
                        expanded_scores.append(anomaly_scores[0].mean().item())
                    else:
                        expanded_scores.append(anomaly_scores[0])
                else:
                    # 해당 시퀀스의 점수 사용
                    seq_idx = i - self.window_size + 1
                    if torch.is_tensor(anomaly_scores[seq_idx]):
                        expanded_scores.append(anomaly_scores[seq_idx].mean().item())
                    else:
                        expanded_scores.append(anomaly_scores[seq_idx])
            
            scores = np.array(expanded_scores)
            
            # 임계값 설정 (95 퍼센타일)
            threshold = np.percentile(scores, 95)
            
            # 이상치 예측
            predictions = np.where(scores > threshold, -1, 1)
        
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
        
        # 시퀀스 생성
        sequences = self._create_sequences(features_scaled)
        
        # 시퀀스가 비어있는 경우 처리
        if len(sequences) == 0:
            # 모든 점수를 0으로 설정
            return np.zeros(len(features))
        
        # 점수 계산
        self.model.eval()
        with torch.no_grad():
            sequences_tensor = torch.FloatTensor(sequences).to(self.device)
            reconstructed, series, prior, sigmas = self.model(sequences_tensor)
            
            # 재구성 오차
            reconstruction_errors = torch.mean((reconstructed - sequences_tensor) ** 2, dim=-1)
            
            # Prior Loss는 복잡하므로 간단하게 처리
            # 재구성 오차만 사용하여 이상 점수 계산
            anomaly_scores = reconstruction_errors
            
            # 시퀀스 길이에 맞춰 점수 확장
            expanded_scores = []
            for i in range(len(features)):
                if i < self.window_size - 1:
                    # 윈도우가 충분하지 않은 경우 첫 번째 점수 사용
                    # anomaly_scores[0]가 텐서인 경우 평균값 사용
                    if torch.is_tensor(anomaly_scores[0]):
                        expanded_scores.append(anomaly_scores[0].mean().item())
                    else:
                        expanded_scores.append(anomaly_scores[0])
                else:
                    # 해당 시퀀스의 점수 사용
                    seq_idx = i - self.window_size + 1
                    if torch.is_tensor(anomaly_scores[seq_idx]):
                        expanded_scores.append(anomaly_scores[seq_idx].mean().item())
                    else:
                        expanded_scores.append(anomaly_scores[seq_idx])
            
            scores = np.array(expanded_scores)
        
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
        axes[1].set_title('Anomaly Transformer 이상 점수')
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
        
        axes[2].set_title('Anomaly Transformer 이상치 탐지 결과')
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
            'Transformer 차원': self.d_model,
            '어텐션 헤드 수': self.n_heads,
            '인코더 레이어 수': self.e_layers
        }
        
        return summary
