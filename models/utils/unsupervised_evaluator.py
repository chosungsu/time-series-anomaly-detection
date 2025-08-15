import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


class UnsupervisedAnomalyEvaluator:
    """
    비지도 이상치 탐지 모델의 성능을 평가하는 클래스
    
    라벨이 없는 상황에서 다양한 방법으로 모델의 성능을 평가합니다:
    1. Proxy-based 평가 (재구성 오차, 군집 분리)
    2. 통계적 분석
    3. 시각적 분석
    4. 도메인 기반 평가
    """
    
    def __init__(self):
        """평가기 초기화"""
        self.evaluation_results = {}
        
    def evaluate_reconstruction_error(self, 
                                   original_data: pd.DataFrame,
                                   reconstructed_data: pd.DataFrame,
                                   anomaly_scores: np.ndarray) -> Dict:
        """
        재구성 오차 기반 평가 (재구성 기반 모델용)
        
        Args:
            original_data: 원본 데이터
            reconstructed_data: 재구성된 데이터
            anomaly_scores: 이상 점수
            
        Returns:
            Dict: 재구성 오차 평가 결과
        """
        # MSE 계산
        mse = np.mean((original_data.values - reconstructed_data.values) ** 2, axis=1)
        
        # 재구성 오차와 이상 점수의 상관관계
        correlation = np.corrcoef(mse, anomaly_scores)[0, 1]
        
        # 재구성 오차 통계
        mse_stats = {
            'mean': np.mean(mse),
            'std': np.std(mse),
            'min': np.min(mse),
            'max': np.max(mse),
            'q25': np.percentile(mse, 25),
            'q75': np.percentile(mse, 75),
            'iqr': np.percentile(mse, 75) - np.percentile(mse, 25)
        }
        
        # 이상 점수와 재구성 오차의 일치성
        # 높은 이상 점수와 높은 재구성 오차가 일치하는지 확인
        high_anomaly = anomaly_scores > np.percentile(anomaly_scores, 90)
        high_reconstruction_error = mse > np.percentile(mse, 90)
        
        consistency = np.sum(high_anomaly & high_reconstruction_error) / np.sum(high_anomaly)
        
        results = {
            'mse_statistics': mse_stats,
            'correlation_with_anomaly_scores': correlation,
            'high_anomaly_consistency': consistency,
            'reconstruction_error': mse
        }
        
        return results
    
    def evaluate_cluster_separation(self, 
                                  data: pd.DataFrame,
                                  anomaly_scores: np.ndarray,
                                  n_clusters: int = 2) -> Dict:
        """
        군집 분리 기반 평가
        
        Args:
            data: 원본 데이터
            anomaly_scores: 이상 점수
            n_clusters: 군집 수
            
        Returns:
            Dict: 군집 분리 평가 결과
        """
        # PCA로 차원 축소 (시각화 및 계산 효율성)
        if data.shape[1] > 10:
            pca = PCA(n_components=10)
            data_reduced = pca.fit_transform(data)
        else:
            data_reduced = data.values
        
        # 이상 점수를 기반으로 2개 군집으로 분할
        threshold = np.percentile(anomaly_scores, 90)  # 상위 10%를 이상치로 간주
        cluster_labels = (anomaly_scores > threshold).astype(int)
        
        # Silhouette Score 계산
        try:
            silhouette = silhouette_score(data_reduced, cluster_labels)
        except:
            silhouette = 0.0
        
        # Davies-Bouldin Index 계산
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans_labels = kmeans.fit_predict(data_reduced)
            davies_bouldin = davies_bouldin_score(data_reduced, kmeans_labels)
        except:
            davies_bouldin = float('inf')
        
        # 군집 간 거리와 군집 내 분산
        cluster_0_indices = np.where(cluster_labels == 0)[0]
        cluster_1_indices = np.where(cluster_labels == 1)[0]
        
        if len(cluster_0_indices) > 0 and len(cluster_1_indices) > 0:
            # 군집 내 분산
            cluster_0_variance = np.var(data_reduced[cluster_0_indices], axis=0).mean()
            cluster_1_variance = np.var(data_reduced[cluster_1_indices], axis=0).mean()
            
            # 군집 간 거리
            cluster_0_center = np.mean(data_reduced[cluster_0_indices], axis=0)
            cluster_1_center = np.mean(data_reduced[cluster_1_indices], axis=0)
            cluster_distance = np.linalg.norm(cluster_0_center - cluster_1_center)
            
            # 분리 지수 (군집 간 거리 / (군집 내 분산의 평균))
            separation_index = cluster_distance / ((cluster_0_variance + cluster_1_variance) / 2)
        else:
            cluster_0_variance = cluster_1_variance = cluster_distance = separation_index = 0.0
        
        results = {
            'silhouette_score': silhouette,
            'davies_bouldin_index': davies_bouldin,
            'cluster_0_variance': cluster_0_variance,
            'cluster_1_variance': cluster_1_variance,
            'cluster_distance': cluster_distance,
            'separation_index': separation_index,
            'cluster_labels': cluster_labels
        }
        
        return results
    
    def evaluate_statistical_properties(self, 
                                      anomaly_scores: np.ndarray,
                                      data: pd.DataFrame) -> Dict:
        """
        이상 점수의 통계적 특성 평가
        
        Args:
            anomaly_scores: 이상 점수
            data: 원본 데이터
            
        Returns:
            Dict: 통계적 특성 평가 결과
        """
        # 이상 점수 통계
        score_stats = {
            'mean': np.mean(anomaly_scores),
            'std': np.std(anomaly_scores),
            'min': np.min(anomaly_scores),
            'max': np.max(anomaly_scores),
            'q25': np.percentile(anomaly_scores, 25),
            'q75': np.percentile(anomaly_scores, 75),
            'iqr': np.percentile(anomaly_scores, 75) - np.percentile(anomaly_scores, 25),
            'skewness': self._calculate_skewness(anomaly_scores),
            'kurtosis': self._calculate_kurtosis(anomaly_scores)
        }
        
        # 이상 점수 분포의 정규성 검정 (Shapiro-Wilk)
        from scipy.stats import shapiro
        try:
            shapiro_stat, shapiro_p = shapiro(anomaly_scores)
            score_stats['shapiro_wilk_statistic'] = shapiro_stat
            score_stats['shapiro_wilk_pvalue'] = shapiro_p
            score_stats['is_normal_distribution'] = shapiro_p > 0.05
        except:
            score_stats['shapiro_wilk_statistic'] = None
            score_stats['shapiro_wilk_pvalue'] = None
            score_stats['is_normal_distribution'] = None
        
        # 이상 점수와 원본 데이터의 상관관계
        correlations = {}
        for col in data.columns:
            try:
                corr = np.corrcoef(anomaly_scores, data[col].values)[0, 1]
                correlations[f'correlation_with_{col}'] = corr
            except:
                correlations[f'correlation_with_{col}'] = 0.0
        
        # 이상 점수의 변화율
        score_changes = np.diff(anomaly_scores)
        change_stats = {
            'mean_change': np.mean(score_changes),
            'std_change': np.std(score_changes),
            'max_change': np.max(np.abs(score_changes)),
            'change_stability': 1.0 / (1.0 + np.std(score_changes))  # 변화가 적을수록 높음
        }
        
        results = {
            'score_statistics': score_stats,
            'correlations_with_features': correlations,
            'change_statistics': change_stats
        }
        
        return results
    
    def evaluate_threshold_sensitivity(self, 
                                     anomaly_scores: np.ndarray,
                                     thresholds: Optional[List[float]] = None) -> Dict:
        """
        임계값 변화에 따른 민감도 분석
        
        Args:
            anomaly_scores: 이상 점수
            thresholds: 테스트할 임계값 리스트
            
        Returns:
            Dict: 임계값 민감도 분석 결과
        """
        if thresholds is None:
            # 기본 임계값: 5%, 10%, 15%, 20%, 25%, 30%, 35%, 40%, 45%, 50%
            thresholds = [np.percentile(anomaly_scores, p) for p in [95, 90, 85, 80, 75, 70, 65, 60, 55, 50]]
        
        sensitivity_results = []
        
        for threshold in thresholds:
            # 해당 임계값에서 이상치로 분류된 데이터 수
            anomalies_detected = np.sum(anomaly_scores > threshold)
            anomaly_ratio = anomalies_detected / len(anomaly_scores)
            
            # 임계값 근처의 데이터 분포
            near_threshold = np.sum(np.abs(anomaly_scores - threshold) < threshold * 0.1)
            near_threshold_ratio = near_threshold / len(anomaly_scores)
            
            sensitivity_results.append({
                'threshold': threshold,
                'anomalies_detected': anomalies_detected,
                'anomaly_ratio': anomaly_ratio,
                'near_threshold_count': near_threshold,
                'near_threshold_ratio': near_threshold_ratio
            })
        
        # 최적 임계값 추정 (Elbow method와 유사)
        # 이상치 비율의 변화율이 급격히 변하는 지점
        ratios = [r['anomaly_ratio'] for r in sensitivity_results]
        ratio_changes = np.diff(ratios)
        optimal_idx = np.argmax(np.abs(ratio_changes))
        optimal_threshold = thresholds[optimal_idx]
        
        results = {
            'threshold_analysis': sensitivity_results,
            'optimal_threshold': optimal_threshold,
            'optimal_threshold_index': optimal_idx,
            'thresholds': thresholds
        }
        
        return results
    
    def evaluate_temporal_consistency(self, 
                                    anomaly_scores: np.ndarray,
                                    window_size: int = 10) -> Dict:
        """
        시간적 일관성 평가 (시계열 데이터용)
        
        Args:
            anomaly_scores: 이상 점수
            window_size: 윈도우 크기
            
        Returns:
            Dict: 시간적 일관성 평가 결과
        """
        # 이동 평균과 표준편차
        moving_mean = pd.Series(anomaly_scores).rolling(window=window_size, center=True).mean().fillna(method='bfill').fillna(method='ffill')
        moving_std = pd.Series(anomaly_scores).rolling(window=window_size, center=True).std().fillna(method='bfill').fillna(method='ffill')
        
        # 이동 평균에서의 편차
        deviation_from_moving_mean = np.abs(anomaly_scores - moving_mean)
        
        # 시간적 안정성 지수 (편차가 작을수록 높음)
        temporal_stability = 1.0 / (1.0 + np.mean(deviation_from_moving_mean))
        
        # 급격한 변화 지점 탐지
        score_changes = np.abs(np.diff(anomaly_scores))
        sudden_changes = score_changes > np.percentile(score_changes, 95)
        sudden_change_ratio = np.sum(sudden_changes) / len(sudden_changes)
        
        # 연속된 이상치 탐지
        high_scores = anomaly_scores > np.percentile(anomaly_scores, 90)
        consecutive_anomalies = self._find_consecutive_anomalies(high_scores)
        
        results = {
            'temporal_stability': temporal_stability,
            'moving_mean': moving_mean.values,
            'moving_std': moving_std.values,
            'deviation_from_moving_mean': deviation_from_moving_mean,
            'sudden_change_ratio': sudden_change_ratio,
            'sudden_changes': sudden_changes,
            'consecutive_anomalies': consecutive_anomalies,
            'max_consecutive_length': max(consecutive_anomalies) if consecutive_anomalies else 0
        }
        
        return results
    
    def evaluate_model_comparison(self, 
                                models_results: Dict[str, Dict],
                                data: pd.DataFrame) -> Dict:
        """
        여러 모델의 결과를 비교 평가
        
        Args:
            models_results: 모델별 결과 딕셔너리
            data: 원본 데이터
            
        Returns:
            Dict: 모델 비교 평가 결과
        """
        comparison_results = {}
        
        for model_name, results in models_results.items():
            if 'anomaly_scores' not in results:
                continue
                
            anomaly_scores = results['anomaly_scores']
            
            # 각 모델에 대한 개별 평가
            cluster_eval = self.evaluate_cluster_separation(data, anomaly_scores)
            stat_eval = self.evaluate_statistical_properties(anomaly_scores, data)
            threshold_eval = self.evaluate_threshold_sensitivity(anomaly_scores)
            temporal_eval = self.evaluate_temporal_consistency(anomaly_scores)
            
            comparison_results[model_name] = {
                'cluster_separation': cluster_eval,
                'statistical_properties': stat_eval,
                'threshold_sensitivity': threshold_eval,
                'temporal_consistency': temporal_eval
            }
        
        # 모델 간 비교 지표
        if len(comparison_results) > 1:
            model_names = list(comparison_results.keys())
            
            # Silhouette Score 비교
            silhouette_scores = [comparison_results[name]['cluster_separation']['silhouette_score'] 
                               for name in model_names]
            best_silhouette_model = model_names[np.argmax(silhouette_scores)]
            
            # 분리 지수 비교
            separation_indices = [comparison_results[name]['cluster_separation']['separation_index'] 
                                for name in model_names]
            best_separation_model = model_names[np.argmax(separation_indices)]
            
            # 시간적 안정성 비교
            temporal_stabilities = [comparison_results[name]['temporal_consistency']['temporal_stability'] 
                                  for name in model_names]
            best_temporal_model = model_names[np.argmax(temporal_stabilities)]
            
            comparison_summary = {
                'best_silhouette_model': best_silhouette_model,
                'best_separation_model': best_separation_model,
                'best_temporal_model': best_temporal_model,
                'silhouette_scores': dict(zip(model_names, silhouette_scores)),
                'separation_indices': dict(zip(model_names, separation_indices)),
                'temporal_stabilities': dict(zip(model_names, temporal_stabilities))
            }
            
            comparison_results['comparison_summary'] = comparison_summary
        
        return comparison_results
    
    def generate_evaluation_report(self, 
                                 evaluation_results: Dict,
                                 save_path: Optional[str] = None) -> str:
        """
        평가 결과를 요약한 리포트 생성
        
        Args:
            evaluation_results: 평가 결과
            save_path: 저장할 파일 경로
            
        Returns:
            str: 생성된 리포트
        """
        report = []
        report.append("=" * 60)
        report.append("비지도 이상치 탐지 모델 평가 리포트")
        report.append("=" * 60)
        report.append("")
        
        # 모델별 결과 요약
        for model_name, results in evaluation_results.items():
            if model_name == 'comparison_summary':
                continue
                
            report.append(f"📊 {model_name} 모델 평가 결과")
            report.append("-" * 40)
            
            # 군집 분리 결과
            cluster = results['cluster_separation']
            report.append(f"군집 분리 성능:")
            report.append(f"  - Silhouette Score: {cluster['silhouette_score']:.4f}")
            report.append(f"  - Davies-Bouldin Index: {cluster['davies_bouldin_index']:.4f}")
            report.append(f"  - 분리 지수: {cluster['separation_index']:.4f}")
            report.append("")
            
            # 통계적 특성
            stats = results['statistical_properties']
            report.append(f"통계적 특성:")
            report.append(f"  - 이상 점수 평균: {stats['score_statistics']['mean']:.4f}")
            report.append(f"  - 이상 점수 표준편차: {stats['score_statistics']['std']:.4f}")
            report.append(f"  - 정규분포 여부: {'예' if stats['score_statistics']['is_normal_distribution'] else '아니오'}")
            report.append("")
            
            # 시간적 일관성
            temporal = results['temporal_consistency']
            report.append(f"시간적 일관성:")
            report.append(f"  - 시간적 안정성: {temporal['temporal_stability']:.4f}")
            report.append(f"  - 급격한 변화 비율: {temporal['sudden_change_ratio']:.2%}")
            report.append(f"  - 최대 연속 이상치 길이: {temporal['max_consecutive_length']}")
            report.append("")
        
        # 모델 비교 결과
        if 'comparison_summary' in evaluation_results:
            summary = evaluation_results['comparison_summary']
            report.append("🏆 모델 비교 결과")
            report.append("-" * 40)
            report.append(f"최고 Silhouette Score: {summary['best_silhouette_model']}")
            report.append(f"최고 분리 지수: {summary['best_separation_model']}")
            report.append(f"최고 시간적 안정성: {summary['best_temporal_model']}")
            report.append("")
            
            report.append("📈 상세 비교 지표:")
            for metric_name, scores in [('Silhouette Score', summary['silhouette_scores']),
                                       ('분리 지수', summary['separation_indices']),
                                       ('시간적 안정성', summary['temporal_stabilities'])]:
                report.append(f"  {metric_name}:")
                for model, score in scores.items():
                    report.append(f"  - {model}: {score:.4f}")
                report.append("")
            
            # 종합 성능 점수 추가 (🏆 모델 비교 결과 섹션 아래에)
            if 'combined_scores' in summary:
                report.append("🏅 종합 성능 점수 (정규화):")
                for model, score in summary['combined_scores'].items():
                    report.append(f"  - {model}: {score:.4f}")
                report.append("")
        
        report.append("💡 평가 지표 해석 가이드")
        report.append("-" * 40)
        report.append("• Silhouette Score: 높을수록 좋음 (최대 1.0)")
        report.append("• Davies-Bouldin Index: 낮을수록 좋음")
        report.append("• 분리 지수: 높을수록 정상/이상치가 잘 분리됨")
        report.append("• 시간적 안정성: 높을수록 안정적")
        report.append("• 급격한 변화 비율: 낮을수록 안정적")
        report.append("")
        report.append("=" * 60)
        
        final_report = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(final_report)
        
        return final_report
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """왜도를 계산합니다."""
        if len(data) < 3:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """첨도를 계산합니다."""
        if len(data) < 4:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _find_consecutive_anomalies(self, anomaly_mask: np.ndarray) -> List[int]:
        """연속된 이상치의 길이를 찾습니다."""
        consecutive_lengths = []
        current_length = 0
        
        for is_anomaly in anomaly_mask:
            if is_anomaly:
                current_length += 1
            else:
                if current_length > 0:
                    consecutive_lengths.append(current_length)
                    current_length = 0
        
        # 마지막에 연속된 이상치가 있는 경우
        if current_length > 0:
            consecutive_lengths.append(current_length)
        
        return consecutive_lengths if consecutive_lengths else [0]
