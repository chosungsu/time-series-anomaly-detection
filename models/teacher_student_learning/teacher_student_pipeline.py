# -*- coding: utf-8 -*-
"""
Teacher-Student 학습 파이프라인
Teacher: OmniAnomaly + Anomaly Transformer 조합 (오프라인, 정확도 우선)
Student: AutoEncoder (온라인, 딥러닝 기반)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import random
import pandas as pd
from typing import Tuple, Optional, Dict, Any
import sys
import os

# 모델 import를 위한 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, '..', '..', 'models')
sys.path.insert(0, models_dir)

# Teacher 모델들 import
from omnianomaly.omnianomaly_model import OmniAnomalyModel
from an_transformer.anomaly_transformer_model import AnomalyTransformerDetector
TEACHER_MODELS_AVAILABLE = True

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# Seed 고정
def set_seed(seed=42):
    """모든 랜덤 시드를 완전히 고정합니다."""
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # PyTorch의 모든 랜덤 시드 고정
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # NumPy의 모든 랜덤 시드 고정
    np.random.seed(seed)
    
    # Python 내장 랜덤 시드 고정
    random.seed(seed)
    
    print(f"Seed {seed}로 모든 랜덤 시드가 고정되었습니다.")

set_seed(200)


def force_deterministic():
    """완전한 결정론적 실행을 강제합니다."""
    import os
    os.environ['PYTHONHASHSEED'] = '0'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # PyTorch 설정
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # NumPy 설정
    np.random.seed(42)
    
    # Python 설정
    random.seed(42)
    
    print("완전한 결정론적 실행이 강제되었습니다.")


# 강제 결정론적 실행 적용
force_deterministic()


class DenoisingTCNAutoEncoder(nn.Module):
    """
    Denoising TCN-AutoEncoder 모델
    TCN 기반 인코더 + 병목 특징에서의 Isolation Forest
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 32, device: str = "cpu"):
        super(DenoisingTCNAutoEncoder, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 모델 초기화 전에 seed 재설정
        set_seed(200)
        
        # TCN 인코더 (3블록: 채널 16→32→32, kernel=3, dilation=1/2/4)
        self.tcn_encoder = nn.ModuleList([
            nn.Conv1d(input_dim, 16, kernel_size=3, padding=1, dilation=1),
            nn.Conv1d(16, 32, kernel_size=3, padding=2, dilation=2),
            nn.Conv1d(32, 32, kernel_size=3, padding=4, dilation=4)
        ])
        
        # 병목 레이어 (bottleneck)
        self.bottleneck = nn.Linear(32, hidden_dim)
        
        # 디코더 (1D transposed conv)
        self.decoder = nn.ModuleList([
            nn.ConvTranspose1d(hidden_dim, 32, kernel_size=3, padding=1, dilation=1),
            nn.ConvTranspose1d(32, 16, kernel_size=3, padding=2, dilation=2),
            nn.ConvTranspose1d(16, input_dim, kernel_size=3, padding=4, dilation=4)
        ])
        
        # 활성화 함수
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
        # Isolation Forest (병목 특징에서)
        self.isolation_forest = None
        
        # 가중치 초기화를 일관되게 설정
        self._initialize_weights()
        
        self.to(device)
    
    def _initialize_weights(self):
        """가중치를 일관되게 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # 입력 형태: (batch_size, features) -> (batch_size, features, 1)
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)
        
        # TCN 인코더
        encoded = x
        for i, conv in enumerate(self.tcn_encoder):
            encoded = self.relu(conv(encoded))
            encoded = self.dropout(encoded)
        
        # 병목 특징 추출 (batch_size, 32, 1) -> (batch_size, 32)
        bottleneck_features = encoded.squeeze(-1)
        bottleneck = self.bottleneck(bottleneck_features)
        
        # 디코더 (batch_size, hidden_dim) -> (batch_size, hidden_dim, 1)
        decoded = bottleneck.unsqueeze(-1)
        for i, conv in enumerate(self.decoder):
            decoded = self.relu(conv(decoded))
            decoded = self.dropout(decoded)
        
        # 출력 형태 맞춤 (batch_size, input_dim, 1) -> (batch_size, input_dim)
        decoded = decoded.squeeze(-1)
        
        return decoded, bottleneck
    
    def add_noise(self, x, noise_std=0.05):
        """가우시안 노이즈 주입 (0.05~0.1)"""
        noise = torch.randn_like(x) * noise_std
        return x + noise
    
    def fit(self, X, epochs: int = 100, lr: float = 0.001):
        """Denoising AutoEncoder 학습"""
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # 데이터를 텐서로 변환
        if isinstance(X, np.ndarray):
            X_tensor = torch.FloatTensor(X).to(self.device)
        else:
            X_tensor = torch.FloatTensor(X.values).to(self.device)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # 노이즈 주입된 입력
            noisy_input = self.add_noise(X_tensor, noise_std=0.05 + 0.05 * (epoch / epochs))
            
            # 순전파
            reconstructed, bottleneck = self.forward(noisy_input)
            
            # 손실 계산 (원본 데이터와 재구성 데이터 비교)
            loss = criterion(reconstructed, X_tensor)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")
        
        # 병목 특징에서 Isolation Forest 학습
        self._train_isolation_forest(X_tensor)
    
    def _train_isolation_forest(self, X_tensor):
        """병목 특징에서 Isolation Forest 학습"""
        self.eval()
        with torch.no_grad():
            # 병목 특징 추출
            _, bottleneck_features = self.forward(X_tensor)
            bottleneck_np = bottleneck_features.cpu().numpy()
            
            # Isolation Forest 학습 전에 seed 재설정
            set_seed(200)
            
            # Isolation Forest 학습
            from sklearn.ensemble import IsolationForest
            self.isolation_forest = IsolationForest(
                contamination=0.05,
                random_state=200,
                n_estimators=100,
                max_samples='auto',
                bootstrap=False  # 일관된 결과를 위해 False로 설정
            )
            self.isolation_forest.fit(bottleneck_np)
            print("Isolation Forest 학습 완료 (병목 특징)")
    
    def predict(self, X):
        """재구성 오차 + IF 점수 기반 이상치 예측"""
        self.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X_tensor = torch.FloatTensor(X).to(self.device)
            else:
                X_tensor = torch.FloatTensor(X.values).to(self.device)
            
            # 재구성 및 병목 특징 추출
            reconstructed, bottleneck = self.forward(X_tensor)
            
            # 재구성 오차 계산
            mse_loss = F.mse_loss(reconstructed, X_tensor, reduction='none')
            recon_scores = torch.mean(mse_loss, dim=1).cpu().numpy()
            
            # Isolation Forest 점수 계산
            bottleneck_np = bottleneck.cpu().numpy()
            if self.isolation_forest is not None:
                if_scores = -self.isolation_forest.decision_function(bottleneck_np)
            else:
                if_scores = np.zeros(len(bottleneck_np))
            
            # 가중 합계 점수 (0.5 * 재구성 오차 + 0.5 * IF 점수)
            combined_scores = 0.5 * recon_scores + 0.5 * if_scores
            
            # 임계값을 95 퍼센타일로 설정
            threshold = np.percentile(combined_scores, 95)
            predictions = (combined_scores > threshold).astype(int)
            
            return predictions, combined_scores
    
    def score_samples(self, X):
        """재구성 오차 + IF 점수 반환"""
        self.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X_tensor = torch.FloatTensor(X).to(self.device)
            else:
                X_tensor = torch.FloatTensor(X.values).to(self.device)
            
            # 재구성 및 병목 특징 추출
            reconstructed, bottleneck = self.forward(X_tensor)
            
            # 재구성 오차 계산
            mse_loss = F.mse_loss(reconstructed, X_tensor, reduction='none')
            recon_scores = torch.mean(mse_loss, dim=1).cpu().numpy()
            
            # Isolation Forest 점수 계산
            bottleneck_np = bottleneck.cpu().numpy()
            if self.isolation_forest is not None:
                if_scores = -self.isolation_forest.decision_function(bottleneck_np)
            else:
                if_scores = np.zeros(len(bottleneck_np))
            
            # 가중 합계 점수
            combined_scores = 0.5 * recon_scores + 0.5 * if_scores
            
            return combined_scores
    
    def detect_anomalies(self, X):
        """이상치 탐지 (기존 모델들과 일관성을 위한 메서드)"""
        return self.predict(X)


class TeacherStudentAnomalyDetector:
    """
    Teacher-Student 이상치 탐지기
    Teacher: OmniAnomaly + Anomaly Transformer
    Student: Denoising TCN-AutoEncoder + Isolation Forest on Bottleneck
    """
    
    def __init__(self, input_dim: int, window_size: int = 7, 
                 contamination: float = 0.05, device: str = "cpu"):
        """
        초기화
        
        Args:
            input_dim: 입력 특성 차원
            window_size: 윈도우 크기
            contamination: 이상치 비율
            device: 디바이스
        """
        # 모델 초기화 전에 seed 재설정
        set_seed(200)
        
        self.input_dim = input_dim
        self.window_size = window_size
        self.contamination = contamination
        self.device = device
        
        # Teacher 모델들 초기화
        if TEACHER_MODELS_AVAILABLE:
            try:
                self.omnianomaly_teacher = OmniAnomalyModel(
                    x_dim=input_dim, 
                    z_dim=16, 
                    hidden_dim=128, 
                    rnn_hidden=128, 
                    rnn_layers=2, 
                    window_size=window_size, 
                    feature_method='combined', 
                    device=device
                )
                self.anomaly_transformer_teacher = AnomalyTransformerDetector(
                    d_model=64, 
                    n_heads=4, 
                    e_layers=3, 
                    window_size=window_size, 
                    feature_method='combined', 
                    epochs=100
                )
                print("Teacher 모델들 초기화 완료")
            except Exception as e:
                print(f"Teacher 모델 초기화 실패: {e}")
                self.omnianomaly_teacher = None
                self.anomaly_transformer_teacher = None
        else:
            self.omnianomaly_teacher = None
            self.anomaly_transformer_teacher = None
        
        # Student 모델
        self.autoencoder_student = DenoisingTCNAutoEncoder(
            input_dim=35 + 1,  # 35개 특성 + 1개 Teacher 점수
            hidden_dim=64,
            device=device
        )
        
        # 스케일러
        self.scaler = StandardScaler()
        
        # 학습 완료 플래그
        self.is_fitted = False
        
        # Teacher 모델들의 예측 결과 저장
        self.teacher_predictions = {}
        self.teacher_scores = {}
        
    def _create_sequences(self, data: np.ndarray) -> np.ndarray:
        """
        시계열 데이터를 윈도우 시퀀스로 변환
        
        Args:
            data: 입력 데이터 (N, features)
            
        Returns:
            윈도우 시퀀스 (N-window_size+1, window_size, features)
        """
        sequences = []
        for i in range(len(data) - self.window_size + 1):
            sequences.append(data[i:i+self.window_size])
        return np.array(sequences)
    
    def _extract_features(self, sequences: np.ndarray) -> np.ndarray:
        """
        시퀀스에서 통계적 특성 추출
        
        Args:
            sequences: 윈도우 시퀀스 (N, window_size, features)
            
        Returns:
            추출된 특성 (N, features)
        """
        features = []
        for seq in sequences:
            # 통계적 특성 추출
            mean_feat = np.mean(seq, axis=0)
            std_feat = np.std(seq, axis=0)
            min_feat = np.min(seq, axis=0)
            max_feat = np.max(seq, axis=0)
            range_feat = max_feat - min_feat
            
            # 결합된 특성
            combined_feat = np.concatenate([
                mean_feat, std_feat, min_feat, max_feat, range_feat
            ])
            features.append(combined_feat)
        
        return np.array(features)
    
    def fit(self, data: pd.DataFrame) -> 'TeacherStudentAnomalyDetector':
        """
        Teacher 모델들을 학습하고 Student로 지식 증류
        
        Args:
            data: 학습 데이터
            
        Returns:
            self
        """
        print("Teacher-Student 모델 학습 시작...")
        
        # 데이터 전처리
        if hasattr(data, 'values'):
            data = data.values
        
        # 윈도우 시퀀스 생성
        sequences = self._create_sequences(data)
        print(f"윈도우 시퀀스 생성 완료: {sequences.shape}")
        
        # 특성 추출
        features = self._extract_features(sequences)
        print(f"특성 추출 완료: {features.shape}")
        
        # 특성 정규화
        features_scaled = self.scaler.fit_transform(features)
        
        # Teacher 모델들의 예측 결과 생성
        if self.omnianomaly_teacher is not None and self.anomaly_transformer_teacher is not None:
            print("실제 Teacher 모델들을 사용하여 예측...")
            try:
                # 데이터를 DataFrame으로 변환 (Teacher 모델들이 DataFrame을 기대)
                if isinstance(data, np.ndarray):
                    data_df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(data.shape[1])])
                else:
                    data_df = data
                
                # OmniAnomaly Teacher 학습 및 예측
                self.omnianomaly_teacher.fit(data_df)
                omnianomaly_scores = self.omnianomaly_teacher.score_samples(data_df)
                self.teacher_scores['omnianomaly'] = omnianomaly_scores
                print("OmniAnomaly Teacher 예측 완료")
                
                # Anomaly Transformer Teacher 학습 및 예측
                self.anomaly_transformer_teacher.fit(data_df)
                anomaly_transformer_scores = self.anomaly_transformer_teacher.score_samples(data_df)
                self.teacher_scores['anomaly_transformer'] = anomaly_transformer_scores
                print("Anomaly Transformer Teacher 예측 완료")
                
            except Exception as e:
                print(f"Teacher 모델 예측 실패, 시뮬레이션 모드로 전환: {e}")
                # 실패 시 시뮬레이션 모드로 전환
                omnianomaly_scores = self._simulate_omnianomaly_scores(sequences)
                self.teacher_scores['omnianomaly'] = omnianomaly_scores
                anomaly_transformer_scores = self._simulate_anomaly_transformer_scores(sequences)
                self.teacher_scores['anomaly_transformer'] = anomaly_transformer_scores
        else:
            print("Teacher 모델들을 사용할 수 없어 시뮬레이션 모드로 실행...")
            # OmniAnomaly 시뮬레이션 (VAE 기반)
            omnianomaly_scores = self._simulate_omnianomaly_scores(sequences)
            self.teacher_scores['omnianomaly'] = omnianomaly_scores
            
            # Anomaly Transformer 시뮬레이션 (Transformer 기반)
            anomaly_transformer_scores = self._simulate_anomaly_transformer_scores(sequences)
            self.teacher_scores['anomaly_transformer'] = anomaly_transformer_scores
        
        # Teacher들의 예측 결과를 Student로 지식 증류
        print("Student 모델 지식 증류 중...")
        self._distill_knowledge_to_student(features_scaled)
        
        self.is_fitted = True
        print("Teacher-Student 모델 학습 완료!")
        
        return self
    
    def _simulate_omnianomaly_scores(self, sequences: np.ndarray) -> np.ndarray:
        """
        OmniAnomaly 점수 시뮬레이션 (VAE 재구성 오차 기반)
        
        Args:
            sequences: 윈도우 시퀀스
            
        Returns:
            이상 점수
        """
        # VAE 재구성 오차 시뮬레이션
        scores = []
        for seq in sequences:
            # 간단한 재구성 오차 계산
            seq_mean = np.mean(seq, axis=0)
            seq_std = np.std(seq, axis=0)
            
            # 정상 패턴에서 벗어난 정도를 점수로 사용
            deviation = np.mean(np.abs(seq - seq_mean) / (seq_std + 1e-8))
            scores.append(deviation)
        
        return np.array(scores)
    
    def _simulate_anomaly_transformer_scores(self, sequences: np.ndarray) -> np.ndarray:
        """
        Anomaly Transformer 점수 시뮬레이션 (Association Discrepancy 기반)
        
        Args:
            sequences: 윈도우 시퀀스
            
        Returns:
            이상 점수
        """
        # Association Discrepancy 시뮬레이션
        scores = []
        for seq in sequences:
            # 시퀀스 내의 시간적 일관성 측정
            temporal_consistency = np.mean([
                np.corrcoef(seq[i], seq[i+1])[0, 1] 
                for i in range(len(seq)-1)
            ])
            
            # 일관성이 낮을수록 이상 점수 높음
            anomaly_score = 1.0 - (temporal_consistency + 1) / 2
            scores.append(anomaly_score)
        
        return np.array(scores)
    
    def _distill_knowledge_to_student(self, features: np.ndarray):
        """
        Teacher들의 지식을 Student로 증류
        
        Args:
            features: 추출된 특성
        """
        # Teacher들의 점수를 결합하여 Student 학습에 사용
        omnianomaly_scores = self.teacher_scores['omnianomaly']
        anomaly_transformer_scores = self.teacher_scores['anomaly_transformer']
        
        # 차원 확인 및 조정
        print(f"Teacher 점수 차원: OmniAnomaly={omnianomaly_scores.shape}, AnomalyTransformer={anomaly_transformer_scores.shape}")
        print(f"Student 특성 차원: {features.shape}")
        
        # Teacher 점수와 Student 특성의 차원을 맞춤
        if len(omnianomaly_scores) != len(features):
            # Teacher 점수를 Student 특성 길이에 맞게 조정
            if len(omnianomaly_scores) > len(features):
                # Teacher 점수가 더 길면 앞부분만 사용
                omnianomaly_scores = omnianomaly_scores[:len(features)]
                anomaly_transformer_scores = anomaly_transformer_scores[:len(features)]
            else:
                # Teacher 점수가 더 짧으면 마지막 값으로 패딩
                padding_length = len(features) - len(omnianomaly_scores)
                omnianomaly_scores = np.pad(omnianomaly_scores, (0, padding_length), mode='edge')
                anomaly_transformer_scores = np.pad(anomaly_transformer_scores, (0, padding_length), mode='edge')
        
        combined_teacher_scores = (omnianomaly_scores + anomaly_transformer_scores) / 2
        
        # Teacher 점수를 특성에 추가
        features_with_teacher = np.column_stack([
            features, 
            combined_teacher_scores.reshape(-1, 1)
        ])
        
        print(f"최종 결합된 특성 차원: {features_with_teacher.shape}")
        
        # Student 모델 학습
        self.autoencoder_student.fit(features_with_teacher)
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        이상치 예측
        
        Args:
            data: 테스트 데이터
            
        Returns:
            이상치 예측 결과 (-1: 이상, 1: 정상)
        """
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다. fit() 메서드를 먼저 호출하세요.")
        
        # 데이터 전처리
        if hasattr(data, 'values'):
            data = data.values
        
        # 윈도우 시퀀스 생성
        sequences = self._create_sequences(data)
        
        # 특성 추출
        features = self._extract_features(sequences)
        
        # 특성 정규화
        features_scaled = self.scaler.transform(features)
        
        # Teacher들의 점수 계산
        omnianomaly_scores = self._simulate_omnianomaly_scores(sequences)
        anomaly_transformer_scores = self._simulate_anomaly_transformer_scores(sequences)
        
        # Teacher 점수를 특성에 추가
        features_with_teacher = np.column_stack([
            features_scaled, 
            ((omnianomaly_scores + anomaly_transformer_scores) / 2).reshape(-1, 1)
        ])
        
        # Student 모델로 예측
        predictions, _ = self.autoencoder_student.detect_anomalies(features_with_teacher)
        
        # 원본 데이터 길이에 맞게 확장
        expanded_predictions = self._expand_predictions(predictions, len(data))
        
        return expanded_predictions
    
    def score_samples(self, data: pd.DataFrame) -> np.ndarray:
        """
        이상 점수 계산
        
        Args:
            data: 테스트 데이터
            
        Returns:
            이상 점수
        """
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다. fit() 메서드를 먼저 호출하세요.")
        
        # 데이터 전처리
        if hasattr(data, 'values'):
            data = data.values
        
        # 윈도우 시퀀스 생성
        sequences = self._create_sequences(data)
        
        # 특성 추출
        features = self._extract_features(sequences)
        
        # 특성 정규화
        features_scaled = self.scaler.transform(features)
        
        # Teacher들의 점수 계산
        omnianomaly_scores = self._simulate_omnianomaly_scores(sequences)
        anomaly_transformer_scores = self._simulate_anomaly_transformer_scores(sequences)
        
        # Teacher 점수를 특성에 추가
        features_with_teacher = np.column_stack([
            features_scaled, 
            ((omnianomaly_scores + anomaly_transformer_scores) / 2).reshape(-1, 1)
        ])
        
        # Student 모델로 점수 계산 (score_samples 사용)
        scores = self.autoencoder_student.score_samples(features_with_teacher)
        
        # 원본 데이터 길이에 맞게 확장
        expanded_scores = self._expand_scores(scores, len(data))
        
        return expanded_scores
    
    def _expand_predictions(self, predictions: np.ndarray, original_length: int) -> np.ndarray:
        """
        윈도우 예측을 원본 시계열 길이에 맞게 확장
        
        Args:
            predictions: 윈도우 예측 결과
            original_length: 원본 시계열 길이
            
        Returns:
            확장된 예측 결과
        """
        expanded = np.zeros(original_length)
        
        # 윈도우 끝 인덱스에 예측 할당
        for i, pred in enumerate(predictions):
            window_end_idx = i + self.window_size - 1
            if window_end_idx < original_length:
                expanded[window_end_idx] = pred
        
        # 나머지 인덱스는 이전 값으로 채움 (0이 아닌 값으로)
        for i in range(original_length):
            if expanded[i] == 0:
                if i > 0:
                    # 이전 값이 0이 아닌 경우에만 사용
                    if expanded[i-1] != 0:
                        expanded[i] = expanded[i-1]
                    else:
                        # 이전 값도 0인 경우, predictions에서 0이 아닌 첫 번째 값 사용
                        non_zero_preds = predictions[predictions != 0]
                        if len(non_zero_preds) > 0:
                            expanded[i] = non_zero_preds[0]
                        else:
                            # 모든 예측이 0인 경우 기본값 사용 (정상)
                            expanded[i] = 1
                else:
                    # 첫 번째 인덱스인 경우, predictions에서 0이 아닌 첫 번째 값 사용
                    non_zero_preds = predictions[predictions != 0]
                    if len(non_zero_preds) > 0:
                        expanded[i] = non_zero_preds[0]
                    else:
                        # 모든 예측이 0인 경우 기본값 사용 (정상)
                        expanded[i] = 1
        
        return expanded
    
    def _expand_scores(self, scores: np.ndarray, original_length: int) -> np.ndarray:
        """
        윈도우 점수를 원본 시계열 길이에 맞게 확장
        
        Args:
            scores: 윈도우 점수
            original_length: 원본 시계열 길이
            
        Returns:
            확장된 점수
        """
        expanded = np.zeros(original_length)
        
        # 윈도우 끝 인덱스에 점수 할당
        for i, score in enumerate(scores):
            window_end_idx = i + self.window_size - 1
            if window_end_idx < original_length:
                expanded[window_end_idx] = score
        
        # 나머지 인덱스는 이전 값으로 채움 (0이 아닌 값으로)
        for i in range(original_length):
            if expanded[i] == 0:
                if i > 0:
                    # 이전 값이 0이 아닌 경우에만 사용
                    if expanded[i-1] != 0:
                        expanded[i] = expanded[i-1]
                    else:
                        # 이전 값도 0인 경우, scores에서 0이 아닌 첫 번째 값 사용
                        non_zero_scores = scores[scores != 0]
                        if len(non_zero_scores) > 0:
                            expanded[i] = non_zero_scores[0]
                        else:
                            # 모든 점수가 0인 경우 기본값 사용
                            expanded[i] = 0.1
                else:
                    # 첫 번째 인덱스인 경우, scores에서 0이 아닌 첫 번째 값 사용
                    non_zero_scores = scores[scores != 0]
                    if len(non_zero_scores) > 0:
                        expanded[i] = non_zero_scores[0]
                    else:
                        # 모든 점수가 0인 경우 기본값 사용
                        expanded[i] = 0.1
        
        return expanded
    
    def detect_anomalies(self, data: pd.DataFrame, threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        이상치 탐지
        
        Args:
            data: 테스트 데이터
            threshold: 임계값 (None이면 자동 설정)
            
        Returns:
            (예측 결과, 이상 점수)
        """
        predictions = self.predict(data)
        scores = self.score_samples(data)
        
        # 임계값이 주어지지 않은 경우 자동 설정
        if threshold is None:
            threshold = np.percentile(scores, (1 - self.contamination) * 100)
        
        # 임계값 기반 이상치 탐지
        anomaly_predictions = (scores > threshold).astype(int)
        
        return anomaly_predictions, scores
    
    def plot_results(self, data: pd.DataFrame, predictions: np.ndarray, 
                    scores: np.ndarray, save_path: Optional[str] = None) -> None:
        """
        결과 시각화
        
        Args:
            data: 원본 데이터
            predictions: 예측 결과
            scores: 이상 점수
            save_path: 저장 경로
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
        axes[1].plot(data.index, scores, 'b-', alpha=0.7, label='Teacher-Student 이상 점수')
        axes[1].set_title('Teacher-Student Learning 이상 점수')
        axes[1].set_xlabel('시간')
        axes[1].set_ylabel('이상 점수')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 이상치 탐지 결과 플롯
        anomaly_indices = np.where(predictions == 1)[0]
        normal_indices = np.where(predictions == 0)[0]
        
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
        
        axes[2].set_title('Teacher-Student Learning 이상치 탐지 결과')
        axes[2].set_xlabel('시간')
        axes[2].set_ylabel('값')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"결과 시각화 저장: {save_path}")
    
    def get_anomaly_summary(self, data: pd.DataFrame, predictions: np.ndarray, 
                           scores: np.ndarray) -> Dict[str, Any]:
        """
        이상치 탐지 결과 요약
        
        Args:
            data: 원본 데이터
            predictions: 예측 결과
            scores: 이상 점수
            
        Returns:
            결과 요약
        """
        total_points = len(predictions)
        normal_points = int(np.sum(predictions == 0))
        anomaly_points = int(np.sum(predictions == 1))
        anomaly_ratio = float(np.mean(predictions))
        
        summary = {
            '총 데이터 포인트': total_points,
            '정상 데이터 포인트': normal_points,
            '이상 데이터 포인트': anomaly_points,
            '이상 비율': f"{anomaly_ratio:.2%}",
            '최대 이상 점수': float(np.max(scores)),
            '최소 이상 점수': float(np.min(scores)),
            '평균 이상 점수': float(np.mean(scores)),
            '이상 점수 표준편차': float(np.std(scores)),
            '사용된 모델': 'Teacher-Student Learning',
            'Teacher 모델': 'OmniAnomaly + Anomaly Transformer',
            'Student 모델': 'Denoising TCN-AutoEncoder + IF on Bottleneck',
            '윈도우 크기': self.window_size,
            '이상치 비율 설정': self.contamination
        }
        
        return summary


if __name__ == "__main__":
    # 테스트 코드
    print("=== Teacher-Student Learning Pipeline 테스트 ===")
    
    # 테스트 시작 전에 seed 재설정
    set_seed(200)
    
    # 가상 데이터 생성
    np.random.seed(42)
    n_samples = 1000
    n_features = 7
    
    # 정상 데이터 (대부분)
    normal_data = np.random.randn(n_samples, n_features) * 0.1
    
    # 이상치 데이터 (일부)
    anomaly_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    normal_data[anomaly_indices] = np.random.randn(len(anomaly_indices), n_features) * 2.0
    
    # DataFrame으로 변환
    columns = [f'feature_{i}' for i in range(n_features)]
    data = pd.DataFrame(normal_data, columns=columns)
    
    print(f"데이터 생성 완료: {data.shape}")
    
    # Teacher-Student 모델 초기화
    detector = TeacherStudentAnomalyDetector(
        input_dim=n_features,
        window_size=10,
        contamination=0.05,
        device="cpu"
    )
    
    # 모델 학습
    detector.fit(data)
    
    # 이상치 탐지
    predictions, scores = detector.detect_anomalies(data)
    
    # 결과 요약
    summary = detector.get_anomaly_summary(data, predictions, scores)
    print("\n=== 이상치 탐지 결과 요약 ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # 결과 시각화
    detector.plot_results(data, predictions, scores)
    
    print("\n테스트 완료!")
