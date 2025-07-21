import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from torch.utils.data import TensorDataset, DataLoader
from .env import make_train_env
import os
import matplotlib.pyplot as plt
import pandas as pd

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class RolloutBuffer:
    def __init__(self):
        self.buffer = list()

    def store(self, transition):
        self.buffer.append(transition)

    def sample(self):
        s, a, r, s_prime, done = map(np.array, zip(*self.buffer))
        self.buffer.clear()
        return (
            torch.FloatTensor(s),
            torch.LongTensor(a).squeeze(-1),  # squeeze to remove extra dimension
            torch.FloatTensor(r).unsqueeze(1),
            torch.FloatTensor(s_prime),
            torch.FloatTensor(done).unsqueeze(1)
        )

    @property
    def size(self):
        return len(self.buffer)
    
class MLPCategoricalPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=(64, 64), activation_fn=F.relu):
        super(MLPCategoricalPolicy, self).__init__()
        self.input_layer = nn.Linear(state_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], action_dim)
        self.activation_fn = activation_fn

    def forward(self, x):
        x = self.activation_fn(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fn(hidden_layer(x))
        x = self.output_layer(x)
        return F.softmax(x, dim=-1)

class MLPGaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=(512, ), activation_fn=F.relu):
        super(MLPGaussianPolicy, self).__init__()
        self.input_layer = nn.Linear(state_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            self.hidden_layers.append(hidden_layer)
        self.mu_layer = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_dims[-1], action_dim)
        self.activation_fn = activation_fn

    def forward(self, x):
        x = self.activation_fn(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fn(hidden_layer(x))

        mu = self.mu_layer(x)
        log_std = torch.tanh(self.log_std_layer(x))

        return mu, log_std.exp()


class MLPStateValue(nn.Module):
    def __init__(self, state_dim, hidden_dims=(64, 64), activation_fn=F.relu):
        super(MLPStateValue, self).__init__()
        self.input_layer = nn.Linear(state_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        self.activation_fn = activation_fn

    def forward(self, x):
        x = self.activation_fn(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fn(hidden_layer(x))
        x = self.output_layer(x)

        return x

class PPO:
    def __init__(
        self,
        policy_type,
        env,
        verbose=1,
        hidden_dims=(64, 64),
        activation_fn=F.relu,
        n_steps=2048,
        n_epochs=10,
        batch_size=64,
        policy_lr=0.0003,
        value_lr=0.0003,
        gamma=0.99,
        lmda=0.95,
        clip_ratio=0.2,
        vf_coef=1.0,
        ent_coef=0.01,
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.env = env
        self.verbose = verbose
        
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        self.policy = MLPCategoricalPolicy(state_dim, action_dim, hidden_dims, activation_fn).to(self.device)
        self.value = MLPStateValue(state_dim, hidden_dims, activation_fn).to(self.device)        
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lmda = lmda
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=value_lr)
        
        self.buffer = RolloutBuffer()
        self.eval_envs = None
        self.eval_log_interval = None

    @torch.no_grad()
    def act(self, s, training=True):
        self.policy.train(training)

        s = torch.as_tensor(s, dtype=torch.float, device=self.device)
        if len(s.shape) == 1:
            s = s.unsqueeze(0)
        
        probs = self.policy(s)
        m = Categorical(probs)
        action = m.sample() if training else torch.argmax(probs, dim=-1)

        return action.cpu().numpy().item()  # scalar 값으로 반환
    
    def learn(self, total_timesteps, log_interval=10):
        print(f"학습 시작: 총 {total_timesteps} 타임스텝")
        obs = self.env.reset()
        episode_reward = 0
        episode_length = 0
        timesteps_so_far = 0
        episode_count = 0
        
        for step in range(total_timesteps):
            action = self.act(obs)
            next_obs, reward, done, info = self.env.step(action)
            
            self.buffer.store((obs, [action], reward, next_obs, done))  # action을 list로 저장
            
            obs = next_obs
            episode_reward += reward
            episode_length += 1
            timesteps_so_far += 1
            
            if done:
                episode_count += 1
                if self.verbose and episode_count % 10 == 0:
                    print(f"Episode {episode_count}: 길이={episode_length}, 보상={episode_reward:.2f}")
                obs = self.env.reset()
                episode_reward = 0
                episode_length = 0
            
            if self.buffer.size >= self.n_steps:
                result = self._learn()
                if self.verbose and step % log_interval == 0:
                    print(f"Step {step}/{total_timesteps} ({step/total_timesteps*100:.1f}%): "
                          f"Policy Loss: {result['policy_loss']:.4f}, "
                          f"Value Loss: {result['value_loss']:.4f}, "
                          f"Entropy: {result['entropy_bonus']:.4f}")
                
                # Evaluation
                if self.eval_envs and step % self.eval_log_interval == 0:
                    print(f"Step {step}: 평가 실행 중...")
                    self._evaluate()
                    print(f"Step {step}: 평가 완료")
        
        print(f"학습 완료: 총 {episode_count} 에피소드, {total_timesteps} 타임스텝")
    
    def _learn(self):
        self.policy.train()
        self.value.train()
        s, a, r, s_prime, done = self.buffer.sample()
        s, a, r, s_prime, done = map(lambda x: x.to(self.device), [s, a, r, s_prime, done])
        
        # GAE 및 log_prob_old 계산
        with torch.no_grad():
            delta = r + (1 - done) * self.gamma * self.value(s_prime) - self.value(s)
            adv = torch.clone(delta)
            ret = torch.clone(r)
            for t in reversed(range(len(r) - 1)):
                adv[t] += (1 - done[t]) * self.gamma * self.lmda * adv[t + 1]
                ret[t] += (1 - done[t]) * self.gamma * ret[t + 1]

            # \pi_{old}(a|s) 로그 확률 값 계산하기
            probs = self.policy(s)
            m = Categorical(probs)
            log_prob_old = m.log_prob(a).unsqueeze(1)
        
        # Training the policy and value network ``n_epochs`` time
        dts = TensorDataset(s, a, ret, adv, log_prob_old)
        loader = DataLoader(dts, batch_size=self.batch_size, shuffle=True)
        for e in range(self.n_epochs):
            value_losses, policy_losses, entropy_bonuses = [], [], []
            for batch in loader:
                s_, a_, ret_, adv_, log_prob_old_ = batch
                # 가치 네트워크의 손실함수 계산
                value = self.value(s_)
                value_loss = F.mse_loss(value, ret_)

                # 정책 네트워크의 손실함수 계산
                probs = self.policy(s_)
                m = Categorical(probs)
                log_prob = m.log_prob(a_).unsqueeze(1)
                
                ratio = (log_prob - log_prob_old_).exp()
                surr1 = adv_ * ratio
                surr2 = adv_ * torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)

                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_bonus = -m.entropy().mean()

                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_bonus
                self.value_optimizer.zero_grad()
                self.policy_optimizer.zero_grad()
                loss.backward()
                self.value_optimizer.step()
                self.policy_optimizer.step()

                value_losses.append(value_loss.item())
                policy_losses.append(policy_loss.item())
                entropy_bonuses.append(-entropy_bonus.item())

        result = {'policy_loss': np.mean(policy_losses),
                  'value_loss': np.mean(value_losses),
                  'entropy_bonus': np.mean(entropy_bonuses)}

        return result
    
    def set_eval(self, eval_envs, eval_log_interval):
        self.eval_envs = eval_envs
        self.eval_log_interval = eval_log_interval
    
    def _evaluate(self):
        for dataset_name, eval_info in self.eval_envs.items():
            env = eval_info['env']
            csv_writer = eval_info['csv_writer']
            
            obs, legal = env.reset()
            total_reward = 0
            results = []
            
            while not env.done:
                if len(legal) == 0:
                    break
                    
                # 현재 상태에서 모든 가능한 액션에 대한 observation 계산
                state_features = env._obs()
                
                # 가장 높은 가치를 가진 legal action 선택
                if len(legal) > 0:
                    legal_features = state_features[legal]
                    values = self.value(torch.FloatTensor(legal_features).to(self.device)).cpu().numpy().flatten()
                    best_legal_idx = np.argmax(values)
                    action = legal[best_legal_idx]
                else:
                    action = legal[0] if legal else 0
                
                obs, legal, reward, done, _ = env.step(action)
                total_reward += reward
                
                if env.count % 10 == 0 and env.count > 0:
                    results.append(total_reward)
            
            # 결과 기록
            if results:
                csv_writer.writerow(results)
                eval_info['csv_file'].flush()
                if self.verbose:
                    print(f"  - {dataset_name}: 총 보상={total_reward:.2f}, 이상 탐지={env.count}")
    
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
        }, path + '.pth')
        if self.verbose:
            print(f"모델 저장됨: {path}.pth")
    
    @classmethod
    def load(cls, path):
        # 간단한 로드 구현 - 실제로는 더 복잡한 로직이 필요할 수 있음
        checkpoint = torch.load(path, map_location='cpu')
        
        # 임시 환경 생성 (기본값들)
        
        temp_env = make_train_env(['./data/sensor_data_with_anomalylabel_isolationforest.csv'])
        
        # PPO 모델 생성
        model = cls('MlpPolicy', temp_env, verbose=0)
        model.policy.load_state_dict(checkpoint['policy_state_dict'])
        model.value.load_state_dict(checkpoint['value_state_dict'])
        
        return model

def evaluate_policy(model, env, n_eval_episodes=1, deterministic=False, use_batch=True, dataset_name="sensor_data"):
    """평가 함수 - evaluate.py에서 사용"""
    total_rewards = []
    all_results = []
    confusion_matrix_data = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
    
    for episode in range(n_eval_episodes):
        obs, legal = env.reset()
        total_reward = 0
        results = []
        episode_predictions = []
        episode_ground_truth = []
        
        print(f"전체 데이터셋 크기: {len(legal)}")
        print(f"전체 이상 수: {np.sum(env._Y == 0)}")
        
        while not env.done:
            if len(legal) == 0:
                break
                
            # 현재 상태에서 모든 가능한 액션에 대한 observation 계산
            state_features = env._obs()
            
            # 가장 높은 가치를 가진 legal action 선택
            if len(legal) > 0:
                legal_features = state_features[legal]
                with torch.no_grad():
                    values = model.value(torch.FloatTensor(legal_features).to(model.device)).cpu().numpy().flatten()
                best_legal_idx = np.argmax(values)
                action = legal[best_legal_idx]
            else:
                action = legal[0] if legal else 0
            
            # 액션을 취하기 전의 ground truth 저장
            ground_truth = env._Y[action]  # 0: anomaly, 1: normal
            
            obs, legal, reward, done, _ = env.step(action)
            total_reward += reward
            
            # 예측 결과 저장 (reward > 0이면 이상 탐지 성공)
            prediction = 1 if reward > 0 else 0  # 1: 이상으로 예측, 0: 정상으로 예측
            
            episode_predictions.append(prediction)
            episode_ground_truth.append(ground_truth)
            
            # 진행상황 출력 (100개마다)
            if env.count % 100 == 0 and env.count > 0:
                print(f"진행상황: {env.count}/{len(env._Y)} ({env.count/len(env._Y)*100:.1f}%) - 현재까지 탐지된 이상: {len(env.anomalies)}")
                results.append(total_reward)
        
        # 마지막 진행상황 출력
        print(f"최종 진행상황: {env.count}/{len(env._Y)} ({env.count/len(env._Y)*100:.1f}%) - 총 탐지된 이상: {len(env.anomalies)}")
        
        # Confusion Matrix 계산
        for pred, true in zip(episode_predictions, episode_ground_truth):
            if pred == 1 and true == 0:  # True Positive (이상을 이상으로 예측)
                confusion_matrix_data['tp'] += 1
            elif pred == 1 and true == 1:  # False Positive (정상을 이상으로 예측)
                confusion_matrix_data['fp'] += 1
            elif pred == 0 and true == 1:  # True Negative (정상을 정상으로 예측)
                confusion_matrix_data['tn'] += 1
            elif pred == 0 and true == 0:  # False Negative (이상을 정상으로 예측)
                confusion_matrix_data['fn'] += 1
        
        total_rewards.append(total_reward)
        all_results.extend(results)
    
    mean_reward = np.mean(total_rewards)
    
    # 성능 지표 계산
    tp, fp, tn, fn = confusion_matrix_data['tp'], confusion_matrix_data['fp'], confusion_matrix_data['tn'], confusion_matrix_data['fn']
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    performance_metrics = {
        'confusion_matrix': confusion_matrix_data,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy,
        'total_predictions': tp + tn + fp + fn,
        'anomalies_detected': tp + fn,  # 실제 이상 수
        'anomalies_found': tp,  # 탐지된 이상 수
    }
    
    # 시각화 함수 호출
    create_anomaly_visualization_direct(env, episode_predictions, episode_ground_truth, dataset_name=dataset_name)
    
    return mean_reward, performance_metrics, all_results

def create_anomaly_visualization_direct(env, predictions, ground_truth, dataset_name="sensor_data"):
    """이상 탐지 결과를 시각화하여 저장 (순환 import 방지를 위해 직접 정의)"""
    try:
        # 데이터 로드 - 데이터셋 이름에서 실제 파일명 추출
        if dataset_name == "sensor_data_with_anomalylabel_isolationforest":
            data_path = f'./data/{dataset_name}.csv'
        else:
            data_path = f'./data/{dataset_name}_with_anomalylabel_isolationforest.csv'
            
        if not os.path.exists(data_path):
            print(f"데이터 파일을 찾을 수 없습니다: {data_path}")
            return
            
        df = pd.read_csv(data_path)
        
        # 원본 데이터에서 이상 레이블 추출 - anomaly_label 컬럼 사용
        if 'anomaly_label' in df.columns:
            original_anomalies = df['anomaly_label'] == 1  # 1이 이상, 0이 정상
            original_anomaly_indices = np.where(original_anomalies)[0]
        else:
            # 기존 방식 (첫 번째 컬럼이 레이블)
            original_anomalies = df.iloc[:, 0] == 'anomaly'
            original_anomaly_indices = np.where(original_anomalies)[0]
        
        # 예측된 이상 인덱스 (env.anomalies에서 실제 탐지된 이상들)
        predicted_anomaly_indices = env.anomalies
        
        # 디버그 정보 출력
        print(f"=== 시각화 디버그 정보 ===")
        print(f"데이터 파일: {data_path}")
        print(f"데이터 크기: {len(df)}")
        print(f"원본 이상 수: {len(original_anomaly_indices)}")
        print(f"예측된 이상 수: {len(predicted_anomaly_indices)}")
        print(f"원본 이상 인덱스 (처음 10개): {original_anomaly_indices[:10]}")
        print(f"원본 이상 인덱스 (마지막 10개): {original_anomaly_indices[-10:]}")
        print(f"예측된 이상 인덱스 (처음 10개): {predicted_anomaly_indices[:10]}")
        print(f"예측된 이상 인덱스 (마지막 10개): {predicted_anomaly_indices[-10:]}")
        print(f"==========================")
        
        # 시각화
        plt.figure(figsize=(15, 8))
        
        # 시계열 데이터 선택 (Pressure 컬럼 우선, 없으면 첫 번째 특성 컬럼)
        if 'Pressure' in df.columns:
            time_series = df['Pressure'].values
            ylabel = 'Pressure'
        else:
            # anomaly_label과 key를 제외한 첫 번째 특성 컬럼 사용
            feature_cols = [col for col in df.columns if col not in ['key', 'anomaly_label', 'anomaly_score']]
            if len(feature_cols) > 0:
                time_series = df[feature_cols[0]].values
                ylabel = feature_cols[0]
            else:
                time_series = np.arange(len(df))
                ylabel = 'Index'
        
        # 메인 시계열 플롯
        plt.plot(time_series, 'b-', alpha=0.7, label='데이터', linewidth=1)
        
        # 원본 이상을 수직선으로 표시 (빨간색 점선)
        if len(original_anomaly_indices) > 0:
            for i, idx in enumerate(original_anomaly_indices):
                if i == 0:
                    plt.axvline(x=idx, color='red', linestyle='--', alpha=0.8, linewidth=1.5, label='원본 이상')
                else:
                    plt.axvline(x=idx, color='red', linestyle='--', alpha=0.8, linewidth=1.5)
        
        # 예측된 이상을 수직선으로 표시 (주황색 실선)
        if len(predicted_anomaly_indices) > 0:
            for i, idx in enumerate(predicted_anomaly_indices):
                if i == 0:
                    plt.axvline(x=idx, color='orange', linestyle='-', alpha=0.9, linewidth=2, label='예측된 이상')
                else:
                    plt.axvline(x=idx, color='orange', linestyle='-', alpha=0.9, linewidth=2)
        
        plt.xlabel('인덱스')
        plt.ylabel(ylabel)
        plt.title(f'이상 탐지 결과 시각화\n원본 이상: {len(original_anomaly_indices)}개, 예측된 이상: {len(predicted_anomaly_indices)}개')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # x축 범위 설정 (모든 데이터가 보이도록)
        plt.xlim(0, len(df) - 1)
        
        # 저장
        output_path = './src/img/predict_anomaly_line.png'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"이상 탐지 시각화 저장 완료: {output_path}")
        print(f"원본 이상 수: {len(original_anomaly_indices)}")
        print(f"예측된 이상 수: {len(predicted_anomaly_indices)}")
        
    except Exception as e:
        print(f"시각화 생성 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

