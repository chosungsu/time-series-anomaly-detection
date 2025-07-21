import sys, os
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from .preprocess import save_anomaly_labels

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class TimeSeriesIsolationForestDetector:
    """
    Isolation Forest를 사용한 시계열 이상탐지
    """
    def __init__(self, window_size=20, params=None):
        self.window_size = window_size
        self.params = params if params is not None else {'n_estimators': 100, 'contamination': 0.01}
        self.model = IsolationForest(**self.params)
    
    def fit_predict(self, df, feature_cols, key_col='key'):
        results = []
        scores = []
        for key in sorted(df[key_col].unique()):
            sub_df = df[df[key_col] == key].reset_index(drop=True)
            X = sub_df[feature_cols].values
            n = len(X)
            key_results = np.ones(n)
            key_scores = np.zeros(n)
            for i in range(self.window_size, n):
                window = X[i-self.window_size:i]
                y_pred = self.model.fit_predict(window)
                if_score = self.model.decision_function(window)[-1]
                key_results[i] = y_pred[-1]
                key_scores[i] = if_score
            results.extend(key_results)
            scores.extend(key_scores)
        return np.array(results), np.array(scores)
        
def run_option_1():
    """옵션 1: rms2 음수값 보정 및 변경 내역 출력, 보정된 데이터 저장"""
    print("=== 옵션 1: rms2 음수값 보정 ===")
    
    # 데이터 로드
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'sensor_data.csv'))
    
    # Vibration_RMS2 컬럼의 음수값을 근처 양수값으로 대체하고 변경 내역 출력
    changed = False
    if 'Vibration_RMS2' in df.columns:
        rms2 = df['Vibration_RMS2'].values.copy()
        negative_indices = []
        original_values = []
        new_values = []
        for i in range(len(rms2)):
            if rms2[i] < 0:
                changed = True
                negative_indices.append(i)
                original_values.append(rms2[i])
                prev_idx = i - 1
                next_idx = i + 1
                replaced = False
                while prev_idx >= 0:
                    if rms2[prev_idx] > 0:
                        rms2[i] = rms2[prev_idx]
                        replaced = True
                        break
                    prev_idx -= 1
                if not replaced:
                    while next_idx < len(rms2):
                        if rms2[next_idx] > 0:
                            rms2[i] = rms2[next_idx]
                            break
                        next_idx += 1
                new_values.append(rms2[i])
        df['Vibration_RMS2'] = rms2
        # 변경 내역 출력
        if negative_indices:
            print("음수였던 Vibration_RMS2 값 보정 내역:")
            for idx, orig, new in zip(negative_indices, original_values, new_values):
                print(f"=> 인덱스 {idx}: 기존값 {orig} → 변경값 {new}")
        else:
            print("음수였던 Vibration_RMS2 값이 없습니다.")
    # 보정된 데이터 저장
    if changed:
        save_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sensor_data_rms2_fixed.csv')
        df.to_csv(save_path, index=False)
        print(f"\n보정된 데이터가 저장되었습니다: {save_path}")


def run_option_2():
    """옵션 2: feature별로 key1~6 데이터를 index를 이어붙여 하나의 시계열로 그리고, key별로 색상/범례를 다르게 하여 저장"""
    print("=== 옵션 2: feature별 key1~6 이어붙인 시계열 시각화 ===")
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'sensor_data.csv'))
    img_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'img')
    os.makedirs(img_dir, exist_ok=True)
    feature_cols = [col for col in df.columns if col != 'key']
    unique_keys = df['key'].unique()
    colors = plt.get_cmap('tab10', len(unique_keys))

    for feature in feature_cols:
        plt.figure(figsize=(16, 7))
        concat_idx = 0
        legend_handles = []
        for i, key in enumerate(unique_keys):
            sub_df = df[df['key'] == key].reset_index(drop=True)
            x = range(concat_idx, concat_idx + len(sub_df))
            line, = plt.plot(x, sub_df[feature].values, label=f'Key {key}', color=colors(i))
            legend_handles.append(line)
            concat_idx += len(sub_df)
        plt.title(f'Distribution of {feature} for All Keys')
        plt.xlabel('Index')
        plt.ylabel('Scaled Value')
        plt.legend(handles=legend_handles)
        save_path = os.path.join(img_dir, f'concat_feature_{feature}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'이미지가 저장되었습니다: {save_path}')


def run_option_3():
    """옵션 3: 이상치 csv 저장"""
    model_option = 'isolationforest'

    DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'sensor_data_rms2_fixed.csv')

    df = pd.read_csv(DATA_PATH)
    save_anomaly_labels(df, model_option, window_size=10)

def run_option_4():
    """옵션 4: Isolation Forest 라벨 기반 Pressure 컬럼 시각화 + 이상 인덱스 라인 표시"""
    print("=== 옵션 4: Isolation Forest 라벨 기반 Pressure 컬럼 시각화 + 이상 인덱스 라인 표시 ===")

    # 데이터 로드 (Isolation Forest 라벨 포함 csv)
    data_path = r"D:\Downloads\swproject-bistelligence\bistelligence\data\sensor_data_with_anomalylabel_isolationforest.csv"
    if not os.path.exists(data_path):
        print(f"데이터 파일이 없습니다: {data_path}")
        return
    df = pd.read_csv(data_path)

    # Pressure 컬럼, target 컬럼 확인
    if 'Pressure' not in df.columns:
        print('Pressure 컬럼이 데이터에 없습니다.')
        return
    if 'anomaly_label' not in df.columns:
        print('target 컬럼이 데이터에 없습니다.')
        return

    # 이상 인덱스 추출 (target==1)
    anomaly_indices = df.index[df['anomaly_label'] == 1].tolist()
    print(f'로드된 이상 인덱스 개수: {len(anomaly_indices)}')

    # img 폴더 경로
    img_dir = os.path.join(os.path.dirname(__file__), 'img')
    os.makedirs(img_dir, exist_ok=True)

    # Pressure 컬럼 시각화
    plt.figure(figsize=(16, 8))
    plt.plot(df.index, df['Pressure'], label='Pressure', color='blue', alpha=0.7, linewidth=1)

    # 이상 인덱스에 세로선 추가
    for i, idx in enumerate(anomaly_indices):
        if idx < len(df):
            if i == 0:
                plt.axvline(x=idx, color='red', linestyle='--', alpha=0.8, linewidth=1.5, label='Detected Anomaly')
            else:
                plt.axvline(x=idx, color='red', linestyle='--', alpha=0.8, linewidth=1.5)

    plt.title('Pressure 시계열 데이터와 Isolation Forest 이상 탐지 결과')
    plt.xlabel('Index')
    plt.ylabel('Pressure Value')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 저장
    save_path = os.path.join(img_dir, 'anomaly_lines.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f'이미지가 저장되었습니다: {save_path}')
    print(f'표시된 이상 인덱스: {anomaly_indices[:10]}...' if len(anomaly_indices) > 10 else f'표시된 이상 인덱스: {anomaly_indices}')

def run_option_5():
    """옵션 5: 데이터 분석 및 이상 탐지 결과 분석"""
    print("=== 옵션 5: 데이터 분석 및 이상 탐지 결과 분석 ===")
    
    # 데이터 로드
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sensor_data_with_anomalylabel_isolationforest.csv')
    if not os.path.exists(data_path):
        print(f"데이터 파일이 없습니다: {data_path}")
        return
    
    df = pd.read_csv(data_path)
    
    # 이상 탐지 결과 분석
    print('\n=== 이상 탐지 결과 분석 ===')
    anomaly_data = df[df['anomaly_label'] == 1]
    normal_data = df[df['anomaly_label'] == 0]
    
    print(f'정상 데이터 수: {len(normal_data)}')
    print(f'이상 데이터 수: {len(anomaly_data)}')

    # 이상 점수 분석
    if 'anomaly_score' in df.columns:
        print('\n=== 이상 점수 분석 ===')
        print(f'이상 점수 평균: {df["anomaly_score"].mean():.4f}')
        print(f'이상 점수 표준편차: {df["anomaly_score"].std():.4f}')
        print(f'이상 점수 최소값: {df["anomaly_score"].min():.4f}')
        print(f'이상 점수 최대값: {df["anomaly_score"].max():.4f}')
        
        # 이상 점수 분포 시각화
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.hist(df['anomaly_score'], bins=50, alpha=0.7, color='blue')
        plt.title('전체 이상 점수 분포')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Frequency')
        
        plt.subplot(2, 2, 2)
        plt.hist(normal_data['anomaly_score'], bins=30, alpha=0.7, color='green', label='정상')
        plt.hist(anomaly_data['anomaly_score'], bins=30, alpha=0.7, color='red', label='이상')
        plt.title('정상/이상 데이터별 이상 점수 분포')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Frequency')
        plt.legend()
        
        plt.subplot(2, 2, 3)
        plt.scatter(df.index, df['anomaly_score'], alpha=0.6, s=1)
        plt.title('이상 점수 시계열')
        plt.xlabel('Index')
        plt.ylabel('Anomaly Score')
        
        plt.subplot(2, 2, 4)
        plt.boxplot([normal_data['anomaly_score'], anomaly_data['anomaly_score']], 
                   labels=['정상', '이상'])
        plt.title('정상/이상 데이터별 이상 점수 박스플롯')
        plt.ylabel('Anomaly Score')
        
        plt.tight_layout()
        
        # 저장
        img_dir = os.path.join(os.path.dirname(__file__), 'img')
        os.makedirs(img_dir, exist_ok=True)
        save_path = os.path.join(img_dir, 'anomaly_score_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f'\n이상 점수 분석 이미지가 저장되었습니다: {save_path}')
    
    print('\n=== 분석 완료 ===')


def run_option_6():
    """옵션 6: 데이터 정규화"""
    print("=== 옵션 6: 데이터 정규화 ===")

    # 데이터 로드
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sensor_data_with_anomalylabel_isolationforest.csv')

    raw_df = pd.read_csv(data_path)
    # key와 anomaly_score 컬럼 제거
    columns_to_drop = ['key', 'anomaly_score']
    df = raw_df.drop(columns=[col for col in columns_to_drop if col in raw_df.columns], errors='ignore')

    # timestamp 컬럼이 없으면 생성 (0부터 순차적으로)
    if 'timestamp' not in df.columns:
        df.insert(0, 'timestamp', range(len(df)))
    else:
        # 이미 있으면 맨 앞으로 이동
        timestamp = df.pop('timestamp')
        df.insert(0, 'timestamp', timestamp)

    feature_cols = ['Pressure', 'Power1', 'Power2', 'Vibration_Peak1', 'Vibration_RMS1', 'Vibration_Peak2', 'Vibration_RMS2']
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    df.to_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'sensor_data_with_anomalylabel_isolationforest_scaled.csv'), index=False)

    print("데이터 정규화 완료")


if __name__ == "__main__":
    # 커맨드 라인 인자 확인
    if len(sys.argv) < 3 or sys.argv[1] != "--opt":
        print("사용법: python -m eda --opt <옵션>")
        print("옵션:")
        print("  1: rms2 음수값 보정")
        print("  2: feature별 key1~6 이어붙인 시계열 시각화")
        print("  3: 이상치 csv 저장")
        print("  4: Pressure 컬럼 시각화 + 이상 인덱스 라인 표시")
        print("  5: 데이터 분석 및 이상 탐지 결과 분석")
        print("  6: 데이터 정규화")
        sys.exit(1)
    
    option = sys.argv[2]
    
    if option == "1":
        run_option_1()
    elif option == "2":
        run_option_2()
    elif option == "3":
        run_option_3()
    elif option == "4":
        run_option_4()
    elif option == "5":
        run_option_5()
    elif option == "6":
        run_option_6()
    else:
        print("잘못된 옵션입니다. 1, 2, 3, 4, 5, 6을 입력해주세요.")
        print("사용법: python -m eda --opt <옵션>")
        print("옵션:")
        print("  1: rms2 음수값 보정")
        print("  2: feature별 key1~6 이어붙인 시계열 시각화")
        print("  3: 이상치 csv 저장")
        print("  4: Pressure 컬럼 시각화 + 이상 인덱스 라인 표시")
        print("  5: 데이터 분석 및 이상 탐지 결과 분석")
        print("  6: 데이터 정규화")
        sys.exit(1)