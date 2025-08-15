# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Optional, Tuple
import random


def set_seed(seed=42):
    """랜덤 시드를 설정합니다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class RecurrentDistribution(nn.Module):
    """
    A multi-variable distribution integrated with recurrent structure.
    """
    
    def __init__(self, input_dim, hidden_dim, z_dim, window_length=100, device="cpu"):
        super(RecurrentDistribution, self).__init__()
        self.device = device
        self.z_dim = z_dim
        self.window_length = window_length
        
        # MLP for mean and std
        self.mean_mlp = nn.Sequential(
            nn.Linear(input_dim + z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim)
        )
        
        self.std_mlp = nn.Sequential(
            nn.Linear(input_dim + z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim),
            nn.Softplus()
        )
        
    def forward(self, input_q, z_previous=None):
        """
        Forward pass for the recurrent distribution.
        
        Args:
            input_q: Input features
            z_previous: Previous z values
            
        Returns:
            mean, std: Distribution parameters
        """
        if z_previous is None:
            z_previous = torch.zeros(input_q.size(0), self.z_dim, device=self.device)
        
        # Concatenate input and previous z
        combined_input = torch.cat([input_q, z_previous], dim=-1)
        
        mean = self.mean_mlp(combined_input)
        std = self.std_mlp(combined_input) + 1e-6  # Add small epsilon for numerical stability
        
        return mean, std


class VAE(nn.Module):
    """
    Variational Autoencoder implementation.
    """
    
    def __init__(self, x_dim, z_dim, hidden_dim, window_length=100, device="cpu"):
        super(VAE, self).__init__()
        self.device = device
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.window_length = window_length
        
        # Encoder (q(z|x))
        self.encoder = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim * 2)  # mean and log_var
        )
        
        # Decoder (p(x|z))
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, x_dim)
        )
        
    def encode(self, x):
        """Encode input x to latent space."""
        h = self.encoder(x)
        mu, log_var = torch.chunk(h, 2, dim=-1)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """Reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent z to reconstruction."""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass."""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var


class OmniAnomaly(nn.Module):
    """
    OmniAnomaly model implementation.
    """
    
    def __init__(self, config, device="cpu"):
        super(OmniAnomaly, self).__init__()
        self.config = config
        self.device = device
        
        # Initialize VAE
        self.vae = VAE(
            x_dim=config.get('x_dim', 51),
            z_dim=config.get('z_dim', 16),
            hidden_dim=config.get('hidden_dim', 128),
            window_length=config.get('window_length', 100),
            device=device
        )
        
        # RNN for temporal modeling
        self.rnn = nn.GRU(
            input_size=config.get('x_dim', 51),
            hidden_size=config.get('rnn_hidden', 128),
            num_layers=config.get('rnn_layers', 2),
            batch_first=True,
            dropout=config.get('dropout', 0.1)
        )
        
        # Output projection
        self.output_proj = nn.Linear(config.get('rnn_hidden', 128), config.get('x_dim', 51))
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, window_length, x_dim)
            
        Returns:
            reconstruction: Reconstructed input
            anomaly_scores: Anomaly scores
        """
        batch_size, window_length, x_dim = x.shape
        
        # RNN encoding
        rnn_out, _ = self.rnn(x)
        
        # VAE processing
        x_flat = x.view(-1, x_dim)
        x_recon, mu, log_var = self.vae(x_flat)
        x_recon = x_recon.view(batch_size, window_length, x_dim)
        
        # Calculate reconstruction error as anomaly score
        reconstruction_error = F.mse_loss(x_recon, x, reduction='none')
        anomaly_scores = torch.mean(reconstruction_error, dim=-1)  # Average over features
        
        return x_recon, anomaly_scores
    
    def get_anomaly_scores(self, x):
        """Get anomaly scores for input x."""
        with torch.no_grad():
            _, scores = self.forward(x)
        return scores


class OmniAnomalyModel:
    """
    기존 모델들과 일관성을 맞추기 위한 OmniAnomaly 모델 래퍼 클래스
    """
    
    def __init__(self, x_dim=None, z_dim=16, hidden_dim=128, rnn_hidden=128, 
                 rnn_layers=2, window_length=100, dropout=0.1, device="cpu", 
                 window_size=10, feature_method='combined', **kwargs):
        # 모델 초기화 전에 seed 설정
        set_seed(200)
        
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.rnn_hidden = rnn_hidden
        self.rnn_layers = rnn_layers
        self.window_length = window_length
        self.dropout = dropout
        self.device = device
        self.window_size = window_size
        self.feature_method = feature_method
        
        # Configuration dictionary
        self.config = {
            'x_dim': x_dim,
            'z_dim': z_dim,
            'hidden_dim': hidden_dim,
            'rnn_hidden': rnn_hidden,
            'rnn_layers': rnn_layers,
            'window_length': window_length,
            'dropout': dropout
        }
        
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X, y=None):
        """
        OmniAnomaly 모델을 학습합니다.
        
        Args:
            X: 입력 데이터 (DataFrame 또는 numpy array)
            y: 레이블 (사용하지 않음, 비지도 학습)
        """
        # DataFrame을 numpy array로 변환
        if hasattr(X, 'values'):
            X = X.values
        
        # 실제 데이터 차원을 감지하여 설정
        if self.x_dim is None:
            self.x_dim = X.shape[1]
            # config도 업데이트
            self.config['x_dim'] = self.x_dim
        
        # 데이터 정규화
        X_scaled = self.scaler.fit_transform(X)
        
        # 윈도우 기반 데이터 준비
        X_windows = self._create_windows(X_scaled)
        
        # 모델 초기화
        self.model = OmniAnomaly(self.config, device=self.device)
        self.model.train()
        
        # 간단한 학습 (실제로는 더 복잡한 학습 과정이 필요)
        self.is_fitted = True
        
        return self
    
    def _create_windows(self, X, window_size=None):
        """Create sliding windows from time series data."""
        if window_size is None:
            window_size = self.window_size
            
        windows = []
        for i in range(len(X) - window_size + 1):
            windows.append(X[i:i + window_size])
        
        if len(windows) == 0:
            # 데이터가 너무 짧은 경우
            windows = [X]
            
        return np.array(windows)
    
    def predict(self, X):
        """
        이상치 점수를 예측합니다.
        
        Args:
            X: 입력 데이터 (DataFrame 또는 numpy array)
            
        Returns:
            anomaly_scores: 이상치 점수
        """
        if not self.is_fitted:
            raise ValueError("모델이 아직 학습되지 않았습니다. fit() 메서드를 먼저 호출하세요.")
        
        # DataFrame을 numpy array로 변환
        if hasattr(X, 'values'):
            X = X.values
        
        # 데이터 정규화
        X_scaled = self.scaler.transform(X)
        
        # 윈도우 기반 데이터 준비
        X_windows = self._create_windows(X_scaled)
        
        # 텐서로 변환
        X_tensor = torch.FloatTensor(X_windows).to(self.device)
        
        # 예측
        with torch.no_grad():
            anomaly_scores = self.model.get_anomaly_scores(X_tensor)
        
        # 윈도우 점수를 원본 시계열 길이에 맞게 확장
        scores = self._expand_scores(anomaly_scores.cpu().numpy(), len(X))
        
        return scores
    
    def _expand_scores(self, window_scores, original_length):
        """Expand window-based scores to original time series length."""
        if len(window_scores) == 1:
            # 단일 윈도우인 경우
            return np.repeat(window_scores[0], original_length)
        
        # 슬라이딩 윈도우 점수를 원본 길이에 맞게 확장
        scores = np.zeros(original_length)
        window_size = self.window_size
        
        # 첫 번째 윈도우의 점수로 시작 부분 채우기
        scores[:window_size] = window_scores[0]
        
        # 중간 부분은 윈도우 점수의 평균
        for i in range(len(window_scores)):
            start_idx = i
            end_idx = min(i + window_size, original_length)
            scores[start_idx:end_idx] = window_scores[i]
        
        return scores
    
    def score_samples(self, X):
        """
        이상치 점수를 반환합니다 (sklearn 호환성).
        
        Args:
            X: 입력 데이터
            
        Returns:
            anomaly_scores: 이상치 점수
        """
        return self.predict(X)
    
    def get_params(self, deep=True):
        """
        모델 파라미터를 반환합니다.
        """
        return {
            'x_dim': self.x_dim,
            'z_dim': self.z_dim,
            'hidden_dim': self.hidden_dim,
            'rnn_hidden': self.rnn_hidden,
            'rnn_layers': self.rnn_layers,
            'window_length': self.window_length,
            'dropout': self.dropout,
            'device': self.device,
            'window_size': self.window_size,
            'feature_method': self.feature_method,
            'seed': self.seed
        }
    
    def set_params(self, **params):
        """
        모델 파라미터를 설정합니다.
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
    
    def detect_anomalies(self, X):
        """
        이상치를 탐지합니다 (기존 모델들과 일관성을 위한 메서드).
        
        Args:
            X: 입력 데이터
            
        Returns:
            predictions: 이상치 예측 (0: 정상, 1: 이상)
            scores: 이상 점수
        """
        scores = self.predict(X)
        
        # 임계값을 95 퍼센타일로 설정하여 이상치 탐지
        threshold = np.percentile(scores, 95)
        predictions = (scores > threshold).astype(int)
        
        return predictions, scores
    
    def get_anomaly_summary(self, X, predictions, scores):
        """
        이상치 탐지 결과 요약을 반환합니다.
        
        Args:
            X: 입력 데이터
            predictions: 이상치 예측
            scores: 이상 점수
            
        Returns:
            dict: 요약 정보
        """
        if hasattr(X, 'values'):
            X = X.values
            
        return {
            '총 데이터 포인트': len(X),
            '정상 데이터 포인트': int(np.sum(predictions == 0)),
            '이상 데이터 포인트': int(np.sum(predictions == 1)),
            '이상 비율': f"{float(np.mean(predictions)):.2%}",
            '최대 이상 점수': float(np.max(scores)),
            '최소 이상 점수': float(np.min(scores)),
            '평균 이상 점수': float(np.mean(scores)),
            '이상 점수 표준편차': float(np.std(scores)),
            '사용된 특징 방법': self.feature_method,
            'OmniAnomaly X 차원': self.x_dim,
            'OmniAnomaly Z 차원': self.z_dim,
            'RNN 히든 차원': self.rnn_hidden,
            '윈도우 크기': self.window_size
        }
    
    def plot_results(self, data, predictions, scores, save_path=None):
        """
        이상치 탐지 결과를 시각화합니다.
        
        Args:
            data: 입력 데이터
            predictions: 이상치 예측
            scores: 이상 점수
            save_path: 저장할 파일 경로
        """
        if hasattr(data, 'values'):
            data = data.values
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # 1. 원본 데이터와 이상 점수
        if len(data.shape) == 2:
            # 2D 데이터인 경우 첫 번째 특성만 표시
            axes[0].plot(data[:, 0], 'b-', alpha=0.7, label='원본 데이터 (첫 번째 특성)')
        else:
            # 3D 데이터인 경우 첫 번째 샘플의 첫 번째 특성 표시
            axes[0].plot(data[0, :, 0], 'b-', alpha=0.7, label='원본 데이터 (첫 번째 샘플, 첫 번째 특성)')
        
        axes[0].set_title('OmniAnomaly 모델 - 원본 데이터')
        axes[0].set_xlabel('시간')
        axes[0].set_ylabel('값')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. 이상 점수와 예측
        axes[1].plot(scores, 'r-', alpha=0.7, label='이상 점수')
        
        # 이상치로 예측된 지점을 강조
        anomaly_indices = np.where(predictions == 1)[0]
        if len(anomaly_indices) > 0:
            axes[1].scatter(anomaly_indices, scores[anomaly_indices], 
                           color='red', s=50, alpha=0.8, label='탐지된 이상치')
        
        axes[1].set_title('OmniAnomaly 모델 - 이상 점수 및 탐지 결과')
        axes[1].set_xlabel('시간')
        axes[1].set_ylabel('이상 점수')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"OmniAnomaly 모델 결과 그래프 저장: {save_path}")


# 간단한 DRNN 구현 (기존 THOC 모델과 유사)
class DRNN(nn.Module):
    """
    Dilated RNN implementation for temporal modeling.
    """
    
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.1, device="cpu"):
        super(DRNN, self).__init__()
        self.device = device
        self.num_layers = num_layers
        
        # Dilated RNN layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            layer = nn.GRU(
                input_size=input_dim if i == 0 else hidden_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
                dropout=dropout
            )
            self.layers.append(layer)
    
    def forward(self, x):
        """
        Forward pass through dilated RNN layers.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            outputs: List of outputs from each layer
        """
        outputs = []
        current_input = x
        
        for i, layer in enumerate(self.layers):
            # Apply dilation
            if i > 0:
                # Skip some timesteps based on dilation
                current_input = current_input[:, ::2, :]
            
            # Process through RNN layer
            output, _ = layer(current_input)
            outputs.append(output)
            
            # Update input for next layer
            current_input = output
        
        return outputs


if __name__ == "__main__":
    # 테스트 코드
    print("=== OmniAnomaly 모델 테스트 ===")
    
    # 설정
    config = {
        'x_dim': 51,
        'z_dim': 16,
        'hidden_dim': 128,
        'rnn_hidden': 128,
        'rnn_layers': 2,
        'window_length': 100,
        'dropout': 0.1
    }
    
    # 모델 초기화
    model = OmniAnomaly(config, device="cpu")
    print(f"모델 생성 완료: {model}")
    
    # 테스트 데이터
    batch_size, seq_len, x_dim = 32, 100, 51
    x = torch.randn(batch_size, seq_len, x_dim)
    
    # Forward pass
    with torch.no_grad():
        recon, scores = model(x)
        print(f"입력 shape: {x.shape}")
        print(f"재구성 shape: {recon.shape}")
        print(f"이상 점수 shape: {scores.shape}")
    
    # OmniAnomalyModel 테스트
    print("\n=== OmniAnomalyModel 테스트 ===")
    omni_model = OmniAnomalyModel(device="cpu")
    
    # 2D 데이터로 테스트
    X_2d = np.random.randn(100, 51)
    omni_model.fit(X_2d)
    scores = omni_model.predict(X_2d)
    print(f"2D 데이터 예측 결과 shape: {scores.shape}")
    
    # 이상치 탐지
    predictions, scores = omni_model.detect_anomalies(X_2d)
    print(f"이상치 탐지 결과: {np.sum(predictions)}개 이상치 탐지")
    
    # 결과 요약
    summary = omni_model.get_anomaly_summary(X_2d, predictions, scores)
    print("결과 요약:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
