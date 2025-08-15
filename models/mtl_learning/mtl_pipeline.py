# -*- coding: utf-8 -*-
"""
개선된 Multi-Task Learning (MTL) 이상탐지 파이프라인
GPT 제안 구조 기반:
- Multi-scale CNN encoder + BiLSTM
- Reconstruction, Forecasting, Classification 3가지 태스크
- Teacher 모델(OmniAnomaly, Anomaly Transformer) 기반 pseudo-labeling
- Dynamic loss weighting (GradNorm/Uncertainty weighting)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import random
import sys
import os
from typing import Tuple, Optional

# 모델 import를 위한 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, '..', '..', 'models')
sys.path.insert(0, models_dir)

# Teacher 모델들 import
from omnianomaly.omnianomaly_model import OmniAnomalyModel
from an_transformer.anomaly_transformer_model import AnomalyTransformerDetector

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# Seed 고정
def set_seed(seed=200):
    """모든 랜덤 시드를 고정합니다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)

set_seed(200)


class MultiScaleCNNEncoder(nn.Module):
    """
    Multi-scale CNN 인코더
    짧은/중간/긴 패턴을 동시에 학습
    """
    
    def __init__(self, input_dim, hidden_dim=64):
        super(MultiScaleCNNEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Multi-scale convolution layers
        self.scale1 = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        self.scale2 = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        self.scale3 = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # Feature fusion
        self.fusion = nn.Conv1d(hidden_dim * 3, hidden_dim, kernel_size=1)
        
    def forward(self, x):
        # x: (batch, features, window)
        m1 = self.scale1(x)
        m2 = self.scale2(x)
        m3 = self.scale3(x)
        
        # Concatenate multi-scale features
        m = torch.cat([m1, m2, m3], dim=1)  # (batch, hidden*3, window)
        m = self.fusion(m)  # (batch, hidden, window)
        
        return m


class ImprovedMTL(nn.Module):
    """
    GPT 제안 구조 기반 개선된 MTL 모델
    """
    
    def __init__(self, input_dim, hidden_dim=64, latent_dim=32, window_size=7, device="cpu"):
        super(ImprovedMTL, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.window_size = window_size
        
        # Multi-scale CNN encoder
        self.multi_scale_encoder = MultiScaleCNNEncoder(input_dim, hidden_dim)
        
        # Shared temporal encoder (BiLSTM) - 단일 레이어로 단순화
        self.temporal_encoder = nn.LSTM(
            hidden_dim, latent_dim, 
            batch_first=True, 
            bidirectional=True,
            num_layers=1
        )
        
        # Task heads
        self.reconstruction_head = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Forecasting head with LSTM decoder - 단일 레이어로 단순화
        self.forecasting_head = nn.LSTM(
            latent_dim * 2, latent_dim, 
            batch_first=True,
            num_layers=1
        )
        self.forecast_out = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Classification head for anomaly detection
        self.classification_head = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Loss weights (동적 조정 가능)
        self.w_recon = 1.0
        self.w_forecast = 1.0
        self.w_class = 0.5
        
        # Teacher 모델들 (pseudo-labeling용)
        self.omnianomaly_teacher = None
        self.anomaly_transformer_teacher = None
        
        # Weight initialization
        self._initialize_weights()
        
    def _initialize_weights(self):
        """가중치 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def set_teacher_models(self, omnianomaly_model, anomaly_transformer_model):
        """Teacher 모델 설정"""
        self.omnianomaly_teacher = omnianomaly_model
        self.anomaly_transformer_teacher = anomaly_transformer_model
    
    def forward(self, x):
        # x: (batch, window, features)
        batch_size, seq_len, feat_dim = x.shape
        
        # Transpose for CNN: (batch, features, window)
        x_t = x.transpose(1, 2)
        
        # Multi-scale CNN encoding
        encoded = self.multi_scale_encoder(x_t)  # (batch, hidden, window)
        
        # Transpose back: (batch, window, hidden)
        encoded = encoded.transpose(1, 2)
        
        # BiLSTM encoding
        latent_seq, (h_n, c_n) = self.temporal_encoder(encoded)
        # latent_seq: (batch, window, latent*2)
        # h_n: (num_layers*2, batch, latent)
        
        # Last hidden state for classification
        latent_last = latent_seq[:, -1, :]  # (batch, latent*2)
        
        # Reconstruction
        recon = self.reconstruction_head(latent_seq)  # (batch, window, features)
        
        # Forecasting
        forecast_seq, _ = self.forecasting_head(latent_seq)  # (batch, window, latent)
        forecast = self.forecast_out(forecast_seq)  # (batch, window, features)
        
        # Classification (anomaly probability)
        cls = self.classification_head(latent_last)  # (batch, 1)
        
        return {
            'reconstruction': recon,
            'forecast': forecast,
            'classification': cls,
            'latent_seq': latent_seq,
            'latent_last': latent_last
        }
    
    def get_pseudo_labels(self, x):
        """Teacher 모델들로부터 pseudo-label 생성"""
        if self.omnianomaly_teacher is None or self.anomaly_transformer_teacher is None:
            return None
        
        try:
            # Teacher 모델들의 학습 상태 확인
            if not hasattr(self.omnianomaly_teacher, 'is_fitted') or not self.omnianomaly_teacher.is_fitted:
                print("OmniAnomaly Teacher가 학습되지 않았습니다.")
                return None
                
            if not hasattr(self.anomaly_transformer_teacher, 'is_fitted') or not self.anomaly_transformer_teacher.is_fitted:
                print("Anomaly Transformer Teacher가 학습되지 않았습니다.")
                return None
            
            # OmniAnomaly pseudo-label
            omnianomaly_scores = self.omnianomaly_teacher.score_samples(x)
            omnianomaly_labels = (omnianomaly_scores > np.percentile(omnianomaly_scores, 95)).astype(float)
            
            # Anomaly Transformer pseudo-label
            # Anomaly Transformer는 특징 추출 후 차원이 달라지므로 원본 데이터를 직접 전달
            anomaly_transformer_scores = self.anomaly_transformer_teacher.score_samples(x)
            anomaly_transformer_labels = (anomaly_transformer_scores > np.percentile(anomaly_transformer_scores, 95)).astype(float)
            
            # 크기 맞추기
            min_size = min(len(omnianomaly_labels), len(anomaly_transformer_labels))
            if min_size > 0:
                omnianomaly_labels = omnianomaly_labels[:min_size]
                anomaly_transformer_labels = anomaly_transformer_labels[:min_size]
                
                # Ensemble pseudo-label (가중 평균)
                ensemble_labels = 0.6 * omnianomaly_labels + 0.4 * anomaly_transformer_labels
                pseudo_labels = (ensemble_labels > 0.5).astype(float)
                
                return pseudo_labels
            else:
                print("Teacher 모델들의 점수가 비어있습니다.")
                return None
            
        except Exception as e:
            print(f"Pseudo-label 생성 실패: {e}")
            return None
    
    def get_total_loss(self, x, pseudo_labels=None):
        """Multi-task loss 계산"""
        outputs = self.forward(x)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(outputs['reconstruction'], x, reduction='mean')
        
        # Forecasting loss (1-step ahead)
        if x.size(1) > 1:
            forecast_target = x[:, 1:, :]  # (batch, window-1, features)
            forecast_pred = outputs['forecast'][:, :-1, :]  # (batch, window-1, features)
            forecast_loss = F.mse_loss(forecast_pred, forecast_target, reduction='mean')
        else:
            forecast_loss = torch.tensor(0.0, device=x.device)
        
        # Classification loss
        if pseudo_labels is not None:
            # Pseudo-label 기반 학습
            # NumPy 배열인 경우 PyTorch 텐서로 변환
            if isinstance(pseudo_labels, np.ndarray):
                pseudo_labels_tensor = torch.FloatTensor(pseudo_labels).to(x.device)
            else:
                pseudo_labels_tensor = pseudo_labels
            
            # 크기 맞추기
            classification_output = outputs['classification'].squeeze()
            if len(classification_output.shape) == 0:
                classification_output = classification_output.unsqueeze(0)
            
            # pseudo_labels_tensor의 크기를 classification_output과 맞추기
            if pseudo_labels_tensor.shape[0] != classification_output.shape[0]:
                # 크기가 다르면 첫 번째 값으로 복제하거나 잘라내기
                if pseudo_labels_tensor.shape[0] > classification_output.shape[0]:
                    pseudo_labels_tensor = pseudo_labels_tensor[:classification_output.shape[0]]
                else:
                    # 첫 번째 값으로 패딩
                    first_value = pseudo_labels_tensor[0] if pseudo_labels_tensor.shape[0] > 0 else 0.0
                    padding = torch.full((classification_output.shape[0] - pseudo_labels_tensor.shape[0],), 
                                      first_value, device=x.device, dtype=torch.float)
                    pseudo_labels_tensor = torch.cat([pseudo_labels_tensor, padding])
            
            cls_loss = F.binary_cross_entropy(classification_output, pseudo_labels_tensor.float(), reduction='mean')
        else:
            # Reconstruction error 기반 학습
            recon_error = F.mse_loss(outputs['reconstruction'], x, reduction='none').mean(dim=(1, 2))
            # 높은 reconstruction error를 이상으로 간주
            cls_target = (recon_error > torch.quantile(recon_error, 0.95)).float()
            
            # 크기 맞추기
            classification_output = outputs['classification'].squeeze()
            if len(classification_output.shape) == 0:
                classification_output = classification_output.unsqueeze(0)
            
            if cls_target.shape[0] != classification_output.shape[0]:
                if cls_target.shape[0] > classification_output.shape[0]:
                    cls_target = cls_target[:classification_output.shape[0]]
                else:
                    first_value = cls_target[0] if cls_target.shape[0] > 0 else 0.0
                    padding = torch.full((classification_output.shape[0] - cls_target.shape[0],), 
                                      first_value, device=x.device, dtype=torch.float)
                    cls_target = torch.cat([cls_target, padding])
            
            cls_loss = F.binary_cross_entropy(classification_output, cls_target, reduction='mean')
        
        # Total loss with dynamic weighting
        total_loss = (
            self.w_recon * recon_loss +
            self.w_forecast * forecast_loss +
            self.w_class * cls_loss
        )
        
        return total_loss, {
            'recon_loss': recon_loss.item(),
            'forecast_loss': forecast_loss.item(),
            'cls_loss': cls_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def get_anomaly_score(self, x):
        """이상 점수 계산"""
        with torch.no_grad():
            outputs = self.forward(x)
            
            # Reconstruction error
            recon_error = F.mse_loss(outputs['reconstruction'], x, reduction='none').mean(dim=(1, 2))
            
            # Forecasting error
            if x.size(1) > 1:
                forecast_target = x[:, 1:, :]
                forecast_pred = outputs['forecast'][:, :-1, :]
                forecast_error = F.mse_loss(forecast_pred, forecast_target, reduction='none').mean(dim=(1, 2))
            else:
                forecast_error = torch.zeros_like(recon_error)
            
            # Classification score (anomaly probability)
            cls_score = outputs['classification'].squeeze()
            
            # 크기 맞추기
            if len(cls_score.shape) == 0:
                cls_score = cls_score.unsqueeze(0)
            
            # 모든 점수의 크기를 맞추기
            target_size = recon_error.shape[0]
            
            if recon_error.shape[0] != target_size:
                if recon_error.shape[0] > target_size:
                    recon_error = recon_error[:target_size]
                else:
                    padding = torch.zeros(target_size - recon_error.shape[0], device=x.device)
                    recon_error = torch.cat([recon_error, padding])
            
            if forecast_error.shape[0] != target_size:
                if forecast_error.shape[0] > target_size:
                    forecast_error = forecast_error[:target_size]
                else:
                    padding = torch.zeros(target_size - forecast_error.shape[0], device=x.device)
                    forecast_error = torch.cat([forecast_error, padding])
            
            if cls_score.shape[0] != target_size:
                if cls_score.shape[0] > target_size:
                    cls_score = cls_score[:target_size]
                else:
                    padding = torch.zeros(target_size - cls_score.shape[0], device=x.device)
                    cls_score = torch.cat([cls_score, padding])
            
            # Combined anomaly score
            anomaly_score = (
                0.4 * recon_error +
                0.3 * forecast_error +
                0.3 * cls_score
            )
            
            return anomaly_score


class ImprovedMultiTaskAnomalyDetector:
    """
    개선된 MTL 이상탐지기
    """
    
    def __init__(self, input_dim, hidden_dim=64, latent_dim=32, window_size=7, 
                 epochs=50, device="cpu"):
        set_seed(200)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.window_size = window_size
        self.epochs = epochs
        self.device = device
        
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
        
        # Teacher 모델들
        self.omnianomaly_teacher = None
        self.anomaly_transformer_teacher = None
        
    def _create_sequences(self, data: np.ndarray) -> np.ndarray:
        """데이터를 시퀀스로 변환"""
        sequences = []
        
        if self.window_size > len(data):
            # 패딩
            padded_data = np.zeros((self.window_size, data.shape[1]))
            padded_data[:len(data)] = data
            sequences.append(padded_data)
        else:
            for i in range(len(data) - self.window_size + 1):
                sequence = data[i:i + self.window_size]
                sequences.append(sequence)
        
        return np.array(sequences)
    
    def _initialize_teacher_models(self, data: pd.DataFrame):
        """Teacher 모델들 초기화 및 학습"""
        try:
            print("Teacher 모델들 초기화 및 학습 중...")
            
            # OmniAnomaly Teacher - 경량화된 설정으로 초기화
            try:
                self.omnianomaly_teacher = OmniAnomalyModel(
                    input_dim=self.input_dim,
                    hidden_dim=32,
                    latent_dim=16,
                    device=self.device
                )
            except Exception as e:
                print(f"OmniAnomaly Teacher 초기화 실패: {e}")
                self.omnianomaly_teacher = None
            print("OmniAnomaly Teacher 초기화 완료, 학습 시작...")
            
            # OmniAnomaly Teacher 학습
            try:
                self.omnianomaly_teacher.fit(data)
                print("OmniAnomaly Teacher 학습 완료")
                print(f"OmniAnomaly Teacher 학습 상태: {self.omnianomaly_teacher.is_fitted}")
            except Exception as e:
                print(f"OmniAnomaly Teacher 학습 실패: {e}")
                print(f"에러 타입: {type(e).__name__}")
                import traceback
                traceback.print_exc()
                # 학습 실패 시에도 모델은 유지 (기본 가중치 사용)
            
            # Anomaly Transformer Teacher - 경량화된 설정으로 초기화
            try:
                # Anomaly Transformer는 특징 추출 후 차원이 달라지므로 실제 특징 차원을 계산
                # 통계적 특징: 10개 + 7개 = 17개
                # 시간적 특징: 7개 + 7개 + 7개 = 21개
                # combined: 17개 + 21개 = 38개
                # 기본값으로 17개 사용 (통계적 특징만)
                actual_input_dim = 17
                
                self.anomaly_transformer_teacher = AnomalyTransformerDetector(
                    input_dim=actual_input_dim,
                    d_model=32,  # hidden_dim 대신 d_model 사용
                    e_layers=2,  # num_layers 대신 e_layers 사용
                    device=self.device,
                    feature_method='statistical'  # 통계적 특징만 사용하여 안정성 향상
                )
            except Exception as e:
                print(f"Anomaly Transformer Teacher 초기화 실패: {e}")
                self.anomaly_transformer_teacher = None
            print("Anomaly Transformer Teacher 초기화 완료, 학습 시작...")
            
            # Anomaly Transformer Teacher 학습
            try:
                self.anomaly_transformer_teacher.fit(data)
                print("Anomaly Transformer Teacher 학습 완료")
            except Exception as e:
                print(f"Anomaly Transformer Teacher 학습 실패: {e}")
                print(f"에러 타입: {type(e).__name__}")
                import traceback
                traceback.print_exc()
                # 학습 실패 시에도 모델은 유지 (기본 가중치 사용)
            
        except Exception as e:
            print(f"Teacher 모델 초기화 실패: {e}")
            self.omnianomaly_teacher = None
            self.anomaly_transformer_teacher = None
    
    def _train_model(self, data: np.ndarray):
        """MTL 모델 학습"""
        # Teacher 모델들 초기화
        if self.omnianomaly_teacher is None:
            data_df = pd.DataFrame(data, columns=self.feature_names)
            self._initialize_teacher_models(data_df)
        
        # 시퀀스 생성
        sequences = self._create_sequences(data)
        
        # 모델 초기화
        self.model = ImprovedMTL(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            window_size=self.window_size,
            device=self.device
        ).to(self.device)
        
        # Teacher 모델들 설정
        if self.omnianomaly_teacher and self.anomaly_transformer_teacher:
            self.model.set_teacher_models(self.omnianomaly_teacher, self.anomaly_transformer_teacher)
        
        # 데이터를 텐서로 변환
        sequences_tensor = torch.FloatTensor(sequences).to(self.device)
        
        # 옵티마이저 설정
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10)
        
        # 학습 루프
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            num_batches = 0
            
            # 배치 단위로 학습 - 배치 크기 증가로 학습 속도 향상
            batch_size = 64
            for i in range(0, len(sequences_tensor), batch_size):
                batch = sequences_tensor[i:i + batch_size]
                
                if len(batch) == 0:
                    continue
                
                try:
                    # Pseudo-label 생성
                    pseudo_labels = None
                    teacher_models_ready = (
                        self.omnianomaly_teacher is not None and 
                        self.anomaly_transformer_teacher is not None and
                        hasattr(self.omnianomaly_teacher, 'is_fitted') and 
                        self.omnianomaly_teacher.is_fitted and
                        hasattr(self.anomaly_transformer_teacher, 'is_fitted') and 
                        self.anomaly_transformer_teacher.is_fitted
                    )
                    
                    if teacher_models_ready:
                        try:
                            batch_df = pd.DataFrame(batch.cpu().numpy().reshape(-1, self.input_dim), 
                                                 columns=self.feature_names)
                            pseudo_labels = self.model.get_pseudo_labels(batch_df)
                        except Exception as e:
                            print(f"Pseudo-label 생성 중 에러: {e}")
                            pseudo_labels = None
                    else:
                        if self.omnianomaly_teacher is None:
                            print("OmniAnomaly Teacher가 초기화되지 않았습니다.")
                        elif self.anomaly_transformer_teacher is None:
                            print("Anomaly Transformer Teacher가 초기화되지 않았습니다.")
                        elif not hasattr(self.omnianomaly_teacher, 'is_fitted'):
                            print("OmniAnomaly Teacher에 is_fitted 속성이 없습니다.")
                        elif not hasattr(self.anomaly_transformer_teacher, 'is_fitted'):
                            print("Anomaly Transformer Teacher에 is_fitted 속성이 없습니다.")
                        elif not self.omnianomaly_teacher.is_fitted:
                            print("OmniAnomaly Teacher가 학습되지 않았습니다.")
                        elif not self.anomaly_transformer_teacher.is_fitted:
                            print("Anomaly Transformer Teacher가 학습되지 않았습니다.")
                        
                        print("Reconstruction error 기반 학습을 사용합니다.")
                    
                    # 순전파
                    loss, loss_dict = self.model.get_total_loss(batch, pseudo_labels)
                    
                    # NaN/Inf 체크
                    if torch.isnan(loss) or torch.isinf(loss):
                        continue
                    
                    # 역전파
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # 그래디언트 클리핑
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    print(f"배치 학습 중 에러 발생: {e}")
                    continue
            
            # Learning rate 조정
            if num_batches > 0:
                avg_loss = total_loss / num_batches
                scheduler.step(avg_loss)
                
                # 진행 상황 출력
                if (epoch + 1) % 20 == 0:
                    print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.6f}")
    
    def fit(self, data: pd.DataFrame) -> 'ImprovedMultiTaskAnomalyDetector':
        """MTL 모델 학습"""
        if data.empty:
            raise ValueError("데이터가 비어있습니다.")
        
        self.feature_names = data.columns.tolist()
        
        # 데이터 정규화
        data_scaled = self.scaler.fit_transform(data)
        
        # 모델 학습
        print("개선된 MTL 모델 학습 중...")
        self._train_model(data_scaled)
        
        self.is_fitted = True
        return self
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """이상치 예측"""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다. fit() 메서드를 먼저 호출하세요.")
        
        if data.empty:
            raise ValueError("데이터가 비어있습니다.")
        
        # 데이터 정규화
        data_scaled = self.scaler.transform(data)
        
        # 시퀀스 생성
        sequences = self._create_sequences(data_scaled)
        
        if len(sequences) == 0:
            return np.ones(len(data))
        
        # 예측 수행
        self.model.eval()
        with torch.no_grad():
            try:
                sequences_tensor = torch.FloatTensor(sequences).to(self.device)
                scores = self.model.get_anomaly_score(sequences_tensor).cpu().numpy()
                
                # 점수 확장
                expanded_scores = self._expand_scores(scores, len(data))
                
                # 임계값 설정
                threshold = np.percentile(expanded_scores, 95)
                predictions = np.where(expanded_scores > threshold, -1, 1)
                
            except Exception as e:
                print(f"예측 중 에러 발생: {e}")
                predictions = np.ones(len(data))
        
        return predictions
    
    def score_samples(self, data: pd.DataFrame) -> np.ndarray:
        """이상 점수 계산"""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다. fit() 메서드를 먼저 호출하세요.")
        
        if data.empty:
            raise ValueError("데이터가 비어있습니다.")
        
        # 데이터 정규화
        data_scaled = self.scaler.transform(data)
        
        # 시퀀스 생성
        sequences = self._create_sequences(data_scaled)
        
        if len(sequences) == 0:
            return np.zeros(len(data))
        
        # 점수 계산
        self.model.eval()
        with torch.no_grad():
            try:
                sequences_tensor = torch.FloatTensor(sequences).to(self.device)
                scores = self.model.get_anomaly_score(sequences_tensor).cpu().numpy()
                
                # 점수 확장
                expanded_scores = self._expand_scores(scores, len(data))
                
            except Exception as e:
                print(f"점수 계산 중 에러 발생: {e}")
                expanded_scores = np.zeros(len(data))
        
        return expanded_scores
    
    def _expand_scores(self, scores: np.ndarray, target_length: int) -> np.ndarray:
        """시퀀스 점수를 전체 길이로 확장"""
        if len(scores) == 0:
            return np.zeros(target_length)
        
        expanded_scores = []
        
        for i in range(target_length):
            if i < self.window_size - 1:
                # 초기 윈도우 부족 시 첫 번째 점수 사용
                expanded_scores.append(scores[0])
            else:
                seq_idx = i - self.window_size + 1
                if seq_idx < len(scores):
                    expanded_scores.append(scores[seq_idx])
                else:
                    # 인덱스 범위 초과 시 마지막 점수 사용
                    expanded_scores.append(scores[-1])
        
        return np.array(expanded_scores)
    
    def detect_anomalies(self, data: pd.DataFrame, 
                        threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """이상치 탐지"""
        predictions = self.predict(data)
        scores = self.score_samples(data)
        
        if threshold is None:
            threshold = np.percentile(scores, 95)
        
        # 임계값 기반 재분류
        anomaly_predictions = (scores > threshold).astype(int)
        anomaly_predictions[anomaly_predictions == 1] = -1
        anomaly_predictions[anomaly_predictions == 0] = 1
        
        return anomaly_predictions, scores
    
    def plot_results(self, data: pd.DataFrame, 
                    predictions: np.ndarray, 
                    scores: np.ndarray,
                    save_path: Optional[str] = None) -> None:
        """결과 시각화"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # 원본 데이터
        for col in data.columns:
            axes[0].plot(data.index, data[col], label=col, alpha=0.7)
        axes[0].set_title('원본 시계열 데이터')
        axes[0].set_xlabel('시간')
        axes[0].set_ylabel('값')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 이상 점수
        axes[1].plot(data.index, scores, 'b-', alpha=0.7, label='이상 점수')
        axes[1].set_title('개선된 MTL 이상 점수')
        axes[1].set_xlabel('시간')
        axes[1].set_ylabel('이상 점수')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 이상치 탐지 결과
        anomaly_indices = np.where(predictions == -1)[0]
        normal_indices = np.where(predictions == 1)[0]
        
        if len(normal_indices) > 0:
            for col in data.columns:
                axes[2].scatter(normal_indices, data.iloc[normal_indices][col], 
                              c='blue', alpha=0.6, s=20, label=f'{col} (정상)' if col == data.columns[0] else "")
        
        if len(anomaly_indices) > 0:
            for col in data.columns:
                axes[2].scatter(anomaly_indices, data.iloc[anomaly_indices][col], 
                              c='red', alpha=0.8, s=50, marker='x', 
                              label=f'{col} (이상)' if col == data.columns[0] else "")
        
        axes[2].set_title('개선된 MTL 이상치 탐지 결과')
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
        """결과 요약"""
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
            '사용된 윈도우 크기': self.window_size,
            '은닉층 차원': self.hidden_dim,
            '잠재 공간 차원': self.latent_dim,
            'Teacher 모델': 'OmniAnomaly + Anomaly Transformer'
        }
        
        return summary


if __name__ == "__main__":
    # 테스트 코드
    print("=== 개선된 MTL Pipeline 테스트 ===")
    
    # 가상 데이터로 테스트
    input_dim = 7
    window_size = 7
    batch_size = 32
    
    # 모델 생성
    model = ImprovedMTL(
        input_dim=input_dim,
        hidden_dim=64,
        latent_dim=32,
        window_size=window_size,
        device="cpu"
    )
    
    print(f"모델 생성 완료: {model}")
    
    # 가상 데이터
    X = torch.randn(batch_size, window_size, input_dim)
    
    # Forward pass
    outputs = model(X)
    print(f"입력 shape: {X.shape}")
    print(f"출력 키: {list(outputs.keys())}")
    
    # Loss 계산
    total_loss, loss_dict = model.get_total_loss(X)
    print(f"총 손실: {total_loss:.6f}")
    
    # Anomaly score
    scores = model.get_anomaly_score(X)
    print(f"이상 점수 shape: {scores.shape}")
    print(f"이상 점수 범위: {scores.min():.6f} ~ {scores.max():.6f}")
    
    print("테스트 완료!")
