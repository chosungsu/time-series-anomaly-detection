from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import os
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from meta_aad.env import make_train_env, make_eval_env
from meta_aad.ppo2 import PPO
from meta_aad.utils import generate_csv_writer

def train():
    print("=== Active Anomaly Detection Training 시작 ===")
    
    # 하드코딩된 설정값들
    train_datasets = ['sensor_data_with_anomalylabel_isolationforest']
    test_datasets = ['sensor_data_with_anomalylabel_isolationforest']
    budget = None
    num_timesteps = 200000
    log_dir = './log'
    eval_interval = 10
    rl_log_interval = 10
    eval_log_interval = 100

    print(f"설정값:")
    print(f"  - 학습 데이터셋: {train_datasets}")
    print(f"  - 테스트 데이터셋: {test_datasets}")
    print(f"  - 예산: {budget}")
    print(f"  - 총 타임스텝: {num_timesteps}")
    print(f"  - 로그 디렉토리: {log_dir}")
    print(f"  - 평가 간격: {eval_log_interval}")

    anomaly_curve_log = os.path.join(log_dir, 'anomaly_curves')
    if not os.path.exists(anomaly_curve_log):
        os.makedirs(anomaly_curve_log)
        print(f"로그 디렉토리 생성: {anomaly_curve_log}")

    # Generate the paths of datasets
    datapaths = []
    for d in train_datasets:
        datapaths.append(os.path.join('./data', d+'.csv'))
    print(f"데이터 경로: {datapaths}")

    # Make the training environment
    print("학습 환경 생성 중...")
    env = make_train_env(datapaths)
    print("학습 환경 생성 완료")
 
    # Make the testing environments
    print("테스트 환경 생성 중...")
    eval_envs = {}
    for d in test_datasets:
        path = os.path.join('./data', d+'.csv')
        output_path = os.path.join(anomaly_curve_log, d+'.csv') 
        csv_file, csv_writer = generate_csv_writer(output_path)
        eval_envs[d] = {'env': make_eval_env(datapath=path, budget=budget),
                        'csv_writer': csv_writer,
                        'csv_file': csv_file,
                        'mean_reward': 0,
                       }
        print(f"  - {d} 테스트 환경 생성 완료")
    print("모든 테스트 환경 생성 완료")

    # Train the model
    print("PPO 모델 초기화 중...")
    model = PPO('MlpPolicy', env, verbose=1)
    print("PPO 모델 초기화 완료")
    
    print("평가 환경 설정 중...")
    model.set_eval(eval_envs, eval_log_interval)
    print("평가 환경 설정 완료")
    
    print(f"모델 학습 시작 (총 {num_timesteps} 타임스텝)...")
    model.learn(total_timesteps=num_timesteps, log_interval=rl_log_interval)
    print("모델 학습 완료")
    
    print("모델 저장 중...")
    model.save(os.path.join(log_dir, 'model'))
    print(f"모델 저장 완료: {os.path.join(log_dir, 'model')}")
    print("=== Training 완료 ===")

if __name__ == "__main__":
    train()