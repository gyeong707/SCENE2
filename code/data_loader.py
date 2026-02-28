import pandas as pd
import os

def load_and_preprocess(file_path, count=None, seed=42, run_type='test', task_type=None):
    """
    파일을 로드하고 샘플링 수행
    task_type: 'plot' | 'character' | 'kobbq' - KoBBQ는 group_cols가 다름
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    print(f"Loading Dataset...: {file_path}")
    try:
        if run_type == 'test':
            df = pd.read_csv(file_path).head(100)
        else: 
            df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"CSV 읽기 실패: {e}")

    # Data Sampling 
    group_cols = ['Source', 'Category', 'ID', 'Version']

    # 존재하는 컬럼만 사용
    group_cols = [c for c in group_cols if c in df.columns]
    if not group_cols:
        raise ValueError(f"샘플링에 필요한 컬럼이 없습니다. (필요: {group_cols})")

    sampled_df = df.groupby(group_cols, group_keys=False).apply(
        lambda x: x.sample(n=count, random_state=seed)
    )

    sampled_df = sampled_df.reset_index(drop=True)

    return sampled_df