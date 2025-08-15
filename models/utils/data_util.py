import pandas as pd
import numpy as np
import torch

# Seed 고정
def set_seed(seed=42):
    """모든 랜덤 시드를 고정합니다."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data(data_path: str) -> pd.DataFrame:
    """
    센서 데이터를 로드합니다.
    
    Args:
        data_path (str): 데이터 파일 경로
        
    Returns:
        pd.DataFrame: 로드된 데이터
    """
    try:
        data = pd.read_csv(data_path)
        print(f"데이터 로드 완료: {data.shape}")
        print(f"컬럼: {list(data.columns)}")
        return data
    except Exception as e:
        print(f"데이터 로드 실패: {e}")
        return None

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    데이터를 전처리합니다.
    
    Args:
        data (pd.DataFrame): 원본 데이터
        
    Returns:
        pd.DataFrame: 전처리된 데이터
    """
    # key 컬럼 제거 (시계열 인덱스로 사용)
    if 'key' in data.columns:
        data = data.drop('key', axis=1)
    
    # 결측값 처리
    data = data.ffill().bfill()
    
    # 데이터 정규화
    data = (data - data.mean()) / data.std()
    
    print(f"전처리 완료: {data.shape}")
    return data