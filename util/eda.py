import sys, os
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
        
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
    else:
        print("잘못된 옵션입니다. 1, 2을 입력해주세요.")
        print("사용법: python -m eda --opt <옵션>")
        print("옵션:")
        print("  1: rms2 음수값 보정")
        print("  2: feature별 key1~6 이어붙인 시계열 시각화")
        sys.exit(1)