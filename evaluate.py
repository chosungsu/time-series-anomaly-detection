from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import os
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from meta_aad.env import make_eval_env
from meta_aad.ppo2 import PPO, evaluate_policy
from meta_aad.utils import generate_csv_writer

def evaluate():
    print("=== Active Anomaly Detection Evaluation 시작 ===")
    
    # 하드코딩된 설정값들
    test_datasets = ['sensor_data_with_anomalylabel_isolationforest']
    budget = None  # 전체 데이터셋을 탐색하도록 None으로 설정
    log_dir = './results'
    model_path = './log/model.pth'
    eval_interval = 10

    print(f"설정값:")
    print(f"  - 테스트 데이터셋: {test_datasets}")
    print(f"  - 예산: {budget}")
    print(f"  - 결과 디렉토리: {log_dir}")
    print(f"  - 모델 경로: {model_path}")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"결과 디렉토리 생성: {log_dir}")

    # Make the testing environments
    print("테스트 환경 생성 중...")
    eval_envs = {}
    for d in test_datasets:
        path = os.path.join('./data', d+'.csv')
        output_path = os.path.join(log_dir, d+'.csv') 
        csv_file, csv_writer = generate_csv_writer(output_path)
        eval_envs[d] = {'env': make_eval_env(datapath=path, budget=budget),
                        'csv_writer': csv_writer,
                        'csv_file': csv_file,
                        'mean_reward': 0,
                       }
        print(f"  - {d} 테스트 환경 생성 완료")
    print("모든 테스트 환경 생성 완료")

    # Load model
    print("모델 로드 중...")
    model = PPO.load(model_path)
    print("모델 로드 완료")

    print("평가 시작...")
    for d in eval_envs:
        print(f'데이터셋: {d}')
        print('=' * 50)
        reward, performance_metrics, results = evaluate_policy(model, eval_envs[d]['env'], n_eval_episodes=1, deterministic=False, use_batch=True, dataset_name=d)
        
        # 기본 결과 출력
        print(f'\n총 보상: {reward:.4f}')
        
        # Confusion Matrix 및 성능 지표 출력
        if performance_metrics:
            cm = performance_metrics['confusion_matrix']
            print(f'\n=== Confusion Matrix ===')
            print(f'True Positive (TP): {cm["tp"]} - 이상을 이상으로 올바르게 탐지')
            print(f'False Positive (FP): {cm["fp"]} - 정상을 이상으로 잘못 탐지')
            print(f'True Negative (TN): {cm["tn"]} - 정상을 정상으로 올바르게 탐지')
            print(f'False Negative (FN): {cm["fn"]} - 이상을 정상으로 잘못 탐지')
            
            print(f'\n=== 성능 지표 ===')
            print(f'정확도 (Accuracy): {performance_metrics["accuracy"]:.4f}')
            print(f'정밀도 (Precision): {performance_metrics["precision"]:.4f}')
            print(f'재현율 (Recall): {performance_metrics["recall"]:.4f}')
            print(f'F1 점수: {performance_metrics["f1_score"]:.4f}')
            print(f'총 예측 수: {performance_metrics["total_predictions"]}')
            print(f'실제 이상 수: {performance_metrics["anomalies_detected"]}')
            print(f'탐지된 이상 수: {performance_metrics["anomalies_found"]}')
            
            # Confusion Matrix를 CSV로 저장
            cm_csv_path = os.path.join(log_dir, f'{d}_confusion_matrix.csv')
            import csv
            with open(cm_csv_path, 'w', newline='') as cm_file:
                cm_writer = csv.writer(cm_file)
                cm_writer.writerow(['', 'pred: true', 'pred: false'])
                cm_writer.writerow(['true: true', cm['tp'], cm['fn']])
                cm_writer.writerow(['true: false', cm['fp'], cm['tn']])
            print(f'Confusion Matrix 저장: {cm_csv_path}')
            
            # 성능 지표를 CSV로 저장
            metrics_csv_path = os.path.join(log_dir, f'{d}_performance_metrics.csv')
            with open(metrics_csv_path, 'w', newline='') as metrics_file:
                metrics_writer = csv.writer(metrics_file)
                metrics_writer.writerow(['metric', 'value'])
                metrics_writer.writerow(['Accuracy', f'{performance_metrics["accuracy"]:.4f}'])
                metrics_writer.writerow(['Precision', f'{performance_metrics["precision"]:.4f}'])
                metrics_writer.writerow(['Recall', f'{performance_metrics["recall"]:.4f}'])
                metrics_writer.writerow(['F1_Score', f'{performance_metrics["f1_score"]:.4f}'])
                metrics_writer.writerow(['Total_Reward', f'{reward:.4f}'])
                metrics_writer.writerow(['Total_Predictions', performance_metrics["total_predictions"]])
                metrics_writer.writerow(['Anomalies_Detected', performance_metrics["anomalies_detected"]])
                metrics_writer.writerow(['Anomalies_Found', performance_metrics["anomalies_found"]])
            print(f'성능 지표 저장: {metrics_csv_path}')

        eval_envs[d]['csv_writer'].writerow(results)
        eval_envs[d]['csv_file'].flush()
        eval_envs[d]['csv_file'].close()
        print(f'결과 저장 완료: {os.path.join(log_dir, d+".csv")}')
        print('')
    
    print("=== Evaluation 완료 ===")

if __name__ == "__main__":
    evaluate()
