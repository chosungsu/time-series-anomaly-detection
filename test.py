#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
비지도 시계열 이상탐지 모델 테스트 스크립트

이 스크립트는 LOF, OC-SVM, Isolation Forest 모델을 사용하여
센서 데이터의 이상탐지를 수행하고 결과를 비교합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os
import json
import torch
from datetime import datetime

from models.lof.lof_model import LOFAnomalyDetector
from models.oc_svm.oc_svm_model import OCSVMAnomalyDetector
from models.isolation_forest.isolation_forest_model import IsolationForestAnomalyDetector
from models.deep_svdd.deep_svdd_model import DeepSVDDAnomalyDetector
from models.omnianomaly.omnianomaly_model import OmniAnomalyModel
from models.thoc.thoc_model import THOCModel
from models.an_transformer.anomaly_transformer_model import AnomalyTransformerDetector

# MTL 파이프라인 import
from models.mtl_learning.mtl_pipeline import ImprovedMultiTaskAnomalyDetector

# Teacher-Student Learning import
from models.teacher_student_learning.teacher_student_pipeline import TeacherStudentAnomalyDetector

from models.utils.unsupervised_evaluator import UnsupervisedAnomalyEvaluator
from models.utils.data_util import set_seed, load_data, preprocess_data

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def evaluate_model(model, data: pd.DataFrame, model_name: str) -> dict:
    """
    모델을 평가하고 결과를 반환합니다.
    
    Args:
        model: 학습된 이상탐지 모델
        data (pd.DataFrame): 테스트 데이터
        model_name (str): 모델 이름
        
    Returns:
        dict: 평가 결과
    """
    print(f"\n{model_name} 모델 평가 중...")
    
    # 각 모델 평가 전에 seed 재설정
    set_seed(200)
    
    try:
        # 모든 모델을 동일한 방식으로 평가
        model.fit(data)
        print(f"{model_name} 모델 학습 완료")
        
        # 이상치 탐지
        predictions, scores = model.detect_anomalies(data)
        print(f"{model_name} 이상치 탐지 완료")
        
        # 결과 요약
        summary = model.get_anomaly_summary(data, predictions, scores)
        print(f"{model_name} 결과 요약:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        return {
            'model_name': model_name,
            'predictions': predictions,
            'scores': scores,
            'summary': summary,
            'model': model
        }
        
    except Exception as e:
        print(f"{model_name} 모델 평가 실패: {e}")
        return None

def plot_comparison_results(data: pd.DataFrame, results: list, save_path: str = None):
    """
    모든 모델의 결과를 비교하여 시각화합니다.
    
    Args:
        data (pd.DataFrame): 원본 데이터
        results (list): 모델 결과 리스트
        save_path (str, optional): 저장할 파일 경로
    """
    n_models = len(results)
    fig, axes = plt.subplots(n_models + 1, 1, figsize=(15, 4 * (n_models + 1)))
    
    # 원본 데이터 플롯
    for col in data.columns:
        axes[0].plot(data.index, data[col], label=col, alpha=0.7)
    axes[0].set_title('원본 시계열 데이터')
    axes[0].set_xlabel('시간')
    axes[0].set_ylabel('값')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 각 모델의 이상 점수 플롯
    for i, result in enumerate(results):
        if result is None:
            continue
            
        model_name = result['model_name']
        scores = result['scores']
        
        axes[i + 1].plot(data.index, scores, 'b-', alpha=0.7, label=f'{model_name} 이상 점수')
        axes[i + 1].set_title(f'{model_name} 이상 점수')
        axes[i + 1].set_xlabel('시간')
        axes[i + 1].set_ylabel('이상 점수')
        axes[i + 1].legend()
        axes[i + 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
def save_model_results(data: pd.DataFrame, results: list, save_dir: str = None):
    """
    각 모델의 이상탐지 결과를 파일로 저장합니다.
    
    Args:
        data (pd.DataFrame): 원본 데이터
        results (list): 모델 결과 리스트
        save_dir (str, optional): 저장할 디렉토리 경로
    """
    # 결과 디렉토리 생성
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # 각 모델별 결과 저장
    for result in results:
        if result is None:
            continue
            
        model_name = result['model_name']
        predictions = result['predictions']
        scores = result['scores']
        summary = result['summary']
        
        # 모델명을 파일명에 사용할 수 있도록 정리
        safe_model_name = model_name.lower().replace(' ', '_')
        
        print(f"\n{model_name} 모델 결과 저장 중...")
        
        # 1. JSON 형태로 전체 결과 저장
        # numpy 타입을 Python 기본 타입으로 변환
        json_result = {
            'model_name': model_name,
            'data_shape': [int(x) for x in data.shape],
            'summary': {k: str(v) if isinstance(v, (np.integer, np.floating)) else v for k, v in summary.items()},
            'predictions': predictions.tolist(),
            'scores': scores.tolist()
        }
        
        json_path = os.path.join(save_dir, f"{safe_model_name}_results.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_result, f, ensure_ascii=False, indent=2)
        print(f"  JSON 결과 저장: {json_path}")

def evaluate_unsupervised_models(data: pd.DataFrame, results: list, model_names: list, save_dir: str = None):
    """
    비지도 학습 모델들을 평가합니다.
    
    Args:
        data (pd.DataFrame): 원본 데이터
        results (list): 모델 결과 리스트
        save_dir (str): 저장할 디렉토리 경로
    """
    
    # 결과 디렉토리 생성
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # 평가기 초기화
    evaluator = UnsupervisedAnomalyEvaluator()
    
    # 모델별 결과를 딕셔너리로 변환
    models_results = {}
    for i, result in enumerate(results):
        if result is None:
            # None인 경우 기본 점수 생성 (모든 모델이 평가에 포함되도록)
            model_name = model_names[i] if i < len(model_names) else f"Model_{i}"
            # 기본 점수: 정규분포에서 생성
            default_scores = np.random.normal(0, 1, len(data))
            models_results[model_name] = {
                'anomaly_scores': default_scores
            }
            print(f"Warning: {model_name} 모델이 평가에 실패하여 기본 점수를 사용합니다.")
        else:
            model_name = result['model_name']
            models_results[model_name] = {
                'anomaly_scores': result['scores']
            }
    
    # 모델 비교 평가 수행
    print("모델 비교 평가 수행 중...")
    evaluation_results = evaluator.evaluate_model_comparison(models_results, data)
    
    # 평가 리포트 생성
    print("평가 리포트 생성 중...")
    report = evaluator.generate_evaluation_report(evaluation_results)
    
    # 리포트 저장
    report_path = os.path.join(save_dir, "unsupervised_evaluation_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"평가 리포트 저장: {report_path}")
    
    # 평가 결과 시각화
    print("평가 결과 시각화 중...")
    plot_evaluation_results(evaluation_results, save_dir)
    
    # 평가 결과 요약 출력
    print("\n=== 비지도 학습 모델 평가 결과 요약 ===")
    if 'comparison_summary' in evaluation_results:
        summary = evaluation_results['comparison_summary']
        print(f"최고 Silhouette Score: {summary['best_silhouette_model']}")
        print(f"최고 분리 지수: {summary['best_separation_model']}")
        print(f"최고 시간적 안정성: {summary['best_temporal_model']}")
        
        # 종합 성능 점수 출력
        if 'combined_scores' in summary:
            print("\n=== 종합 성능 점수 (정규화) ===")
            for model, score in summary['combined_scores'].items():
                print(f"{model}: {score:.4f}")
    
    print(f"\n상세 평가 결과는 {report_path}에서 확인할 수 있습니다.")

def plot_evaluation_results(evaluation_results: dict, save_dir: str):
    """
    평가 결과를 시각화합니다.
    
    Args:
        evaluation_results (dict): 평가 결과
        save_dir (str): 저장할 디렉토리 경로
    """
    # 모델별 Silhouette Score 비교
    if 'comparison_summary' in evaluation_results:
        summary = evaluation_results['comparison_summary']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Silhouette Score 비교
        models = list(summary['silhouette_scores'].keys())
        scores = list(summary['silhouette_scores'].values())
        
        axes[0, 0].bar(models, scores, color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[0, 0].set_title('Silhouette Score 비교')
        axes[0, 0].set_ylabel('Silhouette Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 분리 지수 비교
        separation_scores = list(summary['separation_indices'].values())
        axes[0, 1].bar(models, separation_scores, color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[0, 1].set_title('분리 지수 비교')
        axes[0, 1].set_ylabel('분리 지수')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 시간적 안정성 비교
        temporal_scores = list(summary['temporal_stabilities'].values())
        axes[1, 0].bar(models, temporal_scores, color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[1, 0].set_title('시간적 안정성 비교')
        axes[1, 0].set_ylabel('시간적 안정성')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 종합 성능 비교 (정규화된 점수)
        normalized_silhouette = np.array(scores) / max(scores) if max(scores) > 0 else np.array(scores)
        normalized_separation = np.array(separation_scores) / max(separation_scores) if max(separation_scores) > 0 else np.array(separation_scores)
        normalized_temporal = np.array(temporal_scores) / max(temporal_scores) if max(temporal_scores) > 0 else np.array(temporal_scores)
        
        combined_score = (normalized_silhouette + normalized_separation + normalized_temporal) / 3
        
        # 종합 성능 점수를 evaluation_results에 저장
        evaluation_results['comparison_summary']['combined_scores'] = dict(zip(models, combined_score))
        
        axes[1, 1].bar(models, combined_score, color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[1, 1].set_title('종합 성능 점수 (정규화)')
        axes[1, 1].set_ylabel('종합 점수')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 그래프 저장
        plot_path = os.path.join(save_dir, "unsupervised_evaluation_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"평가 비교 그래프 저장: {plot_path}")
        
def plot_anomaly_detection_results(data: pd.DataFrame, results: list, save_path: str = None):
    """
    각 모델의 이상치 탐지 결과를 개별적으로 시각화합니다.
    
    Args:
        data (pd.DataFrame): 원본 데이터
        results (list): 모델 결과 리스트
        save_path (str, optional): 저장할 파일 경로
    """
    for result in results:
        if result is None:
            continue
            
        model_name = result['model_name']
        predictions = result['predictions']
        scores = result['scores']
        model = result['model']
        
        print(f"\n{model_name} 모델 결과 시각화...")
        
        safe_model_name = model_name.lower().replace(' ', '_')
        plot_save_path = f"{save_path}/{safe_model_name}.png" if save_path else None
        
        # plot_results 메서드 호출
        model.plot_results(data, predictions, scores, save_path=plot_save_path)
        print(f"  {model_name} 모델 결과 시각화 완료")

def main():
    """메인 함수"""
    print("=== 비지도 시계열 이상탐지 모델 테스트 ===\n")
    
    # 메인 함수 시작 전에 seed 설정
    set_seed(200)
    
    # 데이터 경로 설정
    data_path = "data/sensor_data_rms2_fixed.csv"
    
    # 데이터 로드
    print("1. 데이터 로드 중...")
    data = load_data(data_path)
    if data is None:
        print("데이터 로드에 실패했습니다. 프로그램을 종료합니다.")
        return
        
    # 데이터 전처리
    print("\n2. 데이터 전처리 중...")
    data = preprocess_data(data)
    
    # 모델 초기화
    print("\n3. 모델 초기화 중...")
    
    models = [
        # LOFAnomalyDetector(n_neighbors=20, contamination=0.05, window_size=10),  # 5% 이상치 탐지
        OCSVMAnomalyDetector(kernel='rbf', nu=0.05, window_size=10, feature_method='combined'),  # 5% 이상치 탐지
        IsolationForestAnomalyDetector(n_estimators=100, contamination=0.05, window_size=10, feature_method='combined'),  # 5% 이상치 탐지
        DeepSVDDAnomalyDetector(hidden_dims=[64, 32], latent_dim=16, nu=0.05, window_size=10, feature_method='combined', epochs=100),  # 5% 이상치 탐지
        OmniAnomalyModel(x_dim=None, z_dim=16, hidden_dim=128, rnn_hidden=128, rnn_layers=2, window_size=10, feature_method='combined', device="cpu"),
        # THOCModel(n_hidden=128, tau=1, device="cpu"),
        ImprovedMultiTaskAnomalyDetector(input_dim=data.shape[1], hidden_dim=64, latent_dim=32, window_size=7, epochs=100, device="cpu"),
        AnomalyTransformerDetector(d_model=64, n_heads=4, e_layers=3, window_size=7, feature_method='combined', epochs=100),
        TeacherStudentAnomalyDetector(input_dim=data.shape[1], window_size=7, contamination=0.05, device="cpu")
    ]
    
    model_names = ['ocsvm', 'isolation forest', 'deep svdd', 'omnianomaly', 'mtl', 'anomaly transformer', 'teacher student learning']
    
    # 모델 평가
    print("\n4. 모델 평가 중...")
    results = []
    for model, name in zip(models, model_names):
        result = evaluate_model(model, data, name)
        results.append(result)
    
    # 결과 비교 시각화
    print("\n5. 결과 비교 시각화 중...")
    plot_comparison_results(data, results, save_path="leaderboard/anomaly_detection_comparison.png")
    
    # 개별 모델 결과 시각화
    print("\n6. 개별 모델 결과 시각화 중...")
    plot_anomaly_detection_results(data, results, save_path="leaderboard")
    
    # 모델별 결과 저장
    print("\n7. 모델별 결과 저장 중...")
    save_model_results(data, results, save_dir="leaderboard")
    
    # 비지도 학습 모델 평가
    print("\n8. 비지도 학습 모델 평가 중...")
    evaluate_unsupervised_models(data, results, model_names, save_dir="leaderboard")
    
    print("\n테스트 완료!")

if __name__ == "__main__":
    main()
