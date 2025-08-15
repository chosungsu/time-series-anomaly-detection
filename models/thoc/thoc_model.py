import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import random


def set_seed(seed=42):
    """랜덤 시드를 설정합니다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class THOC(nn.Module):
    def __init__(self, C, W, n_hidden, tau=1, device="cpu"):
        super(THOC, self).__init__()
        self.device = device
        self.C, self.W = C, W
        self.L = math.floor(math.log(W, 2)) + 1  # number of DRNN layers
        self.tau = tau
        self.DRNN = DRNN(n_input=C, n_hidden=n_hidden, n_layers=self.L, device=device)

        # 클러스터 수를 동적으로 계산
        self.clusters = nn.ParameterList([
            nn.Parameter(torch.zeros(n_hidden, max(1, self.L*2 - 2*i)))
            for i in range(self.L)
        ])
        
        for c in self.clusters:
            nn.init.xavier_uniform_(c)

        self.cnets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
            ) for _ in range(self.L)  # nn that maps f_bar to f_hat
        ])

        self.MLP = nn.Sequential(
            nn.Linear(n_hidden*2, n_hidden*2),
            nn.ReLU(),
            nn.Linear(n_hidden*2, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden)
        )

        self.TSSnets = nn.ModuleList([
            nn.Linear(n_hidden, C) for _ in range(self.L)
        ])

    def forward(self, X):
        '''
        :param X: (B, W, C)
        :return: anomaly_scores, loss_dict
        '''
        B, W, C = X.shape
        MTF_output = self.DRNN(X)  # Multiscale Temporal Features from dilated RNN.

        # THOC
        L_THOC = 0
        anomaly_scores = torch.zeros(B, device=self.device)
        f_t_bar = MTF_output[0][:, -1, :].unsqueeze(1)
        Ps, Rs = [], []

        for i, cluster in enumerate(self.clusters):
            # Assignment
            eps = 1e-08
            f_norm, c_norm = torch.norm(f_t_bar, dim=-1, keepdim=True), torch.norm(cluster, dim=0, keepdim=True)
            f_norm, c_norm = torch.max(f_norm, eps*torch.ones_like(f_norm, device=self.device)), torch.max(c_norm, eps*torch.ones_like(c_norm, device=self.device))
            cosine_similarity = torch.einsum(
                "Bph,hn->Bpn", f_t_bar / f_norm, cluster/c_norm
            )

            P_ij = F.softmax(cosine_similarity/self.tau, dim=-1)  # (B, num_cluster_{l-1}, num_cluster_{l})
            R_ij = P_ij.squeeze(1) if len(Ps)==0 \
                else torch.einsum("Bp,Bpn->Bn", Rs[i-1], P_ij)
            Ps.append(P_ij)
            Rs.append(R_ij)

            # Update
            c_vectors = self.cnets[i](f_t_bar)  # (B, num_cluster_{l-1}, hidden_dim)
            f_t_bar = torch.einsum(
                "Bnp,Bph->Bnh", P_ij.transpose(-1, -2), c_vectors
            )  # (B, num_cluster_{l}, hidden_dim)

            # fuse last hidden state
            B, num_prev_clusters, num_next_clusters = P_ij.shape
            if i != self.L-1:
                last_hidden_state = MTF_output[i + 1][:, -1, :].unsqueeze(1).repeat(1, num_next_clusters, 1)
                f_t_bar = self.MLP(torch.cat((f_t_bar, last_hidden_state), dim=-1))

            d = 1 - cosine_similarity
            w = R_ij.unsqueeze(1).repeat(1, num_prev_clusters, 1)
            distances = w * d
            anomaly_scores += torch.mean(distances, dim=(1, 2))
            L_THOC += torch.mean(distances)
        anomaly_scores /= len(self.clusters)

        # ORTH
        L_orth = 0
        for cluster in self.clusters:
            c_sq = cluster.T @ cluster  # (K_l, K_l)
            K_l, _ = c_sq.shape
            L_orth += torch.linalg.matrix_norm(c_sq-torch.eye(K_l, device=self.device))
        L_orth /= len(self.clusters)

        # TSS
        L_TSS = 0
        for i, net in enumerate(self.TSSnets):
            src, tgt = MTF_output[i][:, :-2**(i), :], X[:, 2**(i):, :]
            L_TSS += F.mse_loss(net(src), tgt)
        L_TSS /= len(self.TSSnets)

        loss_dict = {
            "L_THOC": L_THOC,
            "L_orth": L_orth,
            "L_TSS": L_TSS,
        }
        return anomaly_scores, loss_dict

    def predict(self, X):
        """
        이상치 탐지를 위한 예측 함수
        """
        with torch.no_grad():
            anomaly_scores, _ = self.forward(X)
        return anomaly_scores

    def get_anomaly_scores(self, X):
        """
        이상 점수를 반환하는 함수
        """
        return self.predict(X)


class THOCModel:
    """
    기존 모델들과 일관성을 맞추기 위한 THOC 모델 래퍼 클래스
    """
    def __init__(self, n_hidden=128, tau=1, device="cpu", window_size=10, feature_method='combined', **kwargs):
        # 모델 초기화 전에 seed 설정
        set_seed(200)
        
        self.n_hidden = n_hidden
        self.tau = tau
        self.device = device
        self.window_size = window_size
        self.feature_method = feature_method
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X, y=None):
        """
        THOC 모델을 학습합니다.
        
        Args:
            X: 입력 데이터 (DataFrame 또는 numpy array)
            y: 레이블 (사용하지 않음, 비지도 학습)
        """
        # DataFrame을 numpy array로 변환
        if hasattr(X, 'values'):
            X = X.values
        
        # 데이터 형태 확인 및 변환
        if len(X.shape) == 2:
            # (n_samples, n_features) -> (n_samples, 1, n_features)
            X = X.reshape(-1, 1, X.shape[1])
        
        # 데이터 정규화
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = self.scaler.fit_transform(X_reshaped)
        X = X_scaled.reshape(original_shape)
        
        # 모델 초기화
        n_samples, n_timesteps, n_features = X.shape
        self.model = THOC(
            C=n_features, 
            W=n_timesteps, 
            n_hidden=self.n_hidden, 
            tau=self.tau, 
            device=self.device
        )
        
        # 모델을 학습 모드로 설정
        self.model.train()
        self.is_fitted = True
        
        return self
    
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
        
        # 데이터 형태 확인 및 변환
        if len(X.shape) == 2:
            # (n_samples, n_features) -> (n_samples, 1, n_features)
            X = X.reshape(-1, 1, X.shape[1])
        
        # 데이터 정규화
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_scaled = self.scaler.transform(X_reshaped)
        X = X_scaled.reshape(original_shape)
        
        # 텐서로 변환
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # 예측
        with torch.no_grad():
            anomaly_scores = self.model.predict(X_tensor)
        
        return anomaly_scores.cpu().numpy()
    
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
            'n_hidden': self.n_hidden,
            'tau': self.tau,
            'device': self.device,
            'window_size': self.window_size,
            'feature_method': self.feature_method
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
            'THOC 히든 차원': self.n_hidden,
            '온도 파라미터': self.tau,
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
        
        axes[0].set_title('THOC 모델 - 원본 데이터')
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
        
        axes[1].set_title('THOC 모델 - 이상 점수 및 탐지 결과')
        axes[1].set_xlabel('시간')
        axes[1].set_ylabel('이상 점수')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"THOC 모델 결과 그래프 저장: {save_path}")


# Dilated RNN code modified from:
# https://github.com/zalandoresearch/pytorch-dilated-rnn/blob/master/drnn.py
class DRNN(nn.Module):
    def __init__(self, n_input, n_hidden, n_layers, dropout=0, cell_type='GRU', batch_first=True, device="cpu"):
        super(DRNN, self).__init__()

        self.dilations = [2 ** i for i in range(n_layers)]
        self.cell_type = cell_type
        self.batch_first = batch_first
        self.device = device

        layers = []
        if self.cell_type == "GRU":
            cell = nn.GRU
        elif self.cell_type == "RNN":
            cell = nn.RNN
        elif self.cell_type == "LSTM":
            cell = nn.LSTM
        else:
            raise NotImplementedError

        for i in range(n_layers):
            if i == 0:
                c = cell(n_input, n_hidden, dropout=dropout)
            else:
                c = cell(n_hidden, n_hidden, dropout=dropout)
            layers.append(c)
        self.cells = nn.Sequential(*layers)

    def forward(self, inputs, hidden=None):
        inputs = inputs.transpose(0, 1) if self.batch_first else inputs
        outputs = []
        for i, (cell, dilation) in enumerate(zip(self.cells, self.dilations)):
            if hidden is None:
                inputs, _ = self.drnn_layer(cell, inputs, dilation)
            else:
                inputs, hidden[i] = self.drnn_layer(cell, inputs, dilation, hidden[i])
            _output = inputs.transpose(0, 1) if self.batch_first else inputs
            outputs.append(_output)
        return outputs

    def drnn_layer(self, cell, inputs, rate, hidden=None):
        n_steps = len(inputs)
        batch_size = inputs[0].size(0)
        hidden_size = cell.hidden_size

        inputs, _ = self._pad_inputs(inputs, n_steps, rate)
        dilated_inputs = self._prepare_inputs(inputs, rate)

        if hidden is None:
            dilated_outputs, hidden = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size)
        else:
            hidden = self._prepare_inputs(hidden, rate)
            dilated_outputs, hidden = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size, hidden=hidden)

        splitted_outputs = self._split_outputs(dilated_outputs, rate)
        outputs = self._unpad_outputs(splitted_outputs, n_steps)

        return outputs, hidden

    def _apply_cell(self, dilated_inputs, cell, batch_size, rate, hidden_size, hidden=None):
        if hidden is None:
            if self.cell_type == 'LSTM':
                c, m = self.init_hidden(batch_size * rate, hidden_size)
                hidden = (c.unsqueeze(0), m.unsqueeze(0))
            else:
                hidden = self.init_hidden(batch_size * rate, hidden_size).unsqueeze(0)

        dilated_outputs, hidden = cell(dilated_inputs, hidden)

        return dilated_outputs, hidden

    def _unpad_outputs(self, splitted_outputs, n_steps):
        return splitted_outputs[:n_steps]

    def _split_outputs(self, dilated_outputs, rate):
        batchsize = dilated_outputs.size(1) // rate

        blocks = [dilated_outputs[:, i * batchsize: (i + 1) * batchsize, :] for i in range(rate)]

        interleaved = torch.stack((blocks)).transpose(1, 0).contiguous()
        interleaved = interleaved.view(dilated_outputs.size(0) * rate,
                                       batchsize,
                                       dilated_outputs.size(2))
        return interleaved

    def _pad_inputs(self, inputs, n_steps, rate):
        is_even = (n_steps % rate) == 0

        if not is_even:
            dilated_steps = n_steps // rate + 1

            zeros_ = torch.zeros(dilated_steps * rate - inputs.size(0),
                                 inputs.size(1),
                                 inputs.size(2)).to(self.device)

            inputs = torch.cat((inputs, zeros_))
        else:
            dilated_steps = n_steps // rate

        return inputs, dilated_steps

    def _prepare_inputs(self, inputs, rate):
        dilated_inputs = torch.cat([inputs[j::rate, :, :] for j in range(rate)], 1)
        return dilated_inputs

    def init_hidden(self, batch_size, hidden_dim):
        hidden = torch.zeros(batch_size, hidden_dim).to(self.device)
        if self.cell_type == "LSTM":
            memory = torch.zeros(batch_size, hidden_dim).to(self.device)
            return (hidden, memory)
        else:
            return hidden


if __name__ == "__main__":
    B, L, C = 64, 12, 51
    ip = torch.randn((B, L, C))
    drnn = DRNN(n_input=C, n_hidden=128, n_layers=3, dropout=0, cell_type='GRU', batch_first=True, device="cpu")

    print(drnn)
    print(len(drnn(ip)))
    
    # THOC 모델 테스트
    thoc = THOC(C=C, W=L, n_hidden=128, device="cpu")
    anomaly_scores, loss_dict = thoc(ip)
    print(f"Anomaly scores shape: {anomaly_scores.shape}")
    print(f"Loss dict: {loss_dict}")
    
    # THOCModel 테스트
    print("\n=== THOCModel 테스트 ===")
    thoc_model = THOCModel(n_hidden=128, device="cpu")
    
    # 2D 데이터로 테스트
    X_2d = np.random.randn(100, 51)
    thoc_model.fit(X_2d)
    scores = thoc_model.predict(X_2d)
    print(f"2D 데이터 예측 결과 shape: {scores.shape}")
    
    # 3D 데이터로 테스트
    X_3d = np.random.randn(100, 12, 51)
    thoc_model.fit(X_3d)
    scores = thoc_model.predict(X_3d)
    print(f"3D 데이터 예측 결과 shape: {scores.shape}")
