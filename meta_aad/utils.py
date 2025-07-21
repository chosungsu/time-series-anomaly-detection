import numpy as np
import pandas as pd

def read_csv(file, header=None, sep=',', index_col=None, skiprows=None, usecols=None, encoding='utf8'):
    """
    Loads data from a CSV
    """
    data_df = pd.read_csv(file, header=0, sep=sep, index_col=index_col, skiprows=skiprows, usecols=usecols, encoding=encoding)
    return data_df

def dataframe_to_matrix(df, labelindex=0, startcol=1):
    """ 
    Converts a python dataframe in the expected anomaly dataset format to numpy arrays.
    """
    # key와 anomaly_score 컬럼 제거
    if 'key' in df.columns:
        df = df.drop('key', axis=1)
    if 'anomaly_score' in df.columns:
        df = df.drop('anomaly_score', axis=1)
    
    # anomaly_label 컬럼을 찾아서 타겟으로 사용
    if 'anomaly_label' in df.columns:
        label_col = 'anomaly_label'
        feature_cols = [col for col in df.columns if col != label_col]
    else:
        # 기존 방식 (첫 번째 컬럼이 레이블)
        feature_cols = [col for col in df.columns if col != df.columns[labelindex]]
    
    # 특성 데이터 추출
    x = df[feature_cols].values
    
    # 레이블 데이터 추출
    if 'anomaly_label' in df.columns:
        labels = df[label_col].values
        # anomaly_label이 1이면 이상(0), 0이면 정상(1)으로 변환
        labels = np.array([0 if label == 1 else 1 for label in labels], dtype=int)
    else:
        # 기존 방식
        labels = np.array([0 if df.iloc[i, labelindex] == "anomaly" or df.iloc[i, labelindex] == 1 else 1 for i in range(df.shape[0])], dtype=int)
    
    return x, labels


def read_data_as_matrix(datapath):
    """ 
    Reads data from CSV file and returns numpy matrix.
    """
    with open(datapath, "r") as f:
        data = read_csv(f, sep=',')
    X_train, labels = dataframe_to_matrix(data)
    anomalies = np.where(labels == 0)[0]
    return X_train, labels, anomalies
    
def rank_scores(scores):
    """ Rank the scores
    """
    rank = np.argsort(scores)
    return rank
    
def evaluate(e, agent, interval=10):
    """ Evaluate the performance
    """
    s, legal = e.reset()
    done = False
    total_r = 0
    results = []
    counter = 0
    while not done:
        a = agent.eval_step(s, legal)
        s, legal, r, done, _ = e.step(a)
        total_r += r
        counter += 1
        if counter % interval == 0:
            results.append(total_r)
    return total_r, results

def remove_illegal(values, legal):
    """ Zero out the illegal entry
    """
    new_values = np.zeros(values.shape[0])
    new_values[legal] = values[legal]
    return new_values

def run_iforest(X):
    """ Predict the anomaly score with iForest
    """
    from sklearn.ensemble import IsolationForest
    clf = IsolationForest()
    clf.fit(X)
    scores = clf.decision_function(X)

    return scores

def generate_csv_writer(csv_path):
    import csv
    csv_file = open(csv_path, 'w') 
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([str(10*i) for i in range(1, 11)])
    csv_file.flush()
    return csv_file, csv_writer
