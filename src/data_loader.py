import pandas as pd
from pathlib import Path

def load_data(file_path, delimiter=',', encoding='utf-8'):
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        return pd.read_csv(file_path, delimiter=delimiter, encoding=encoding)
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing file: {e}")

if __name__ == '__main__':
    data = load_data(
        r'c:\Users\karbo\OneDrive\Desktop\Data Science\Cust_Seg\data\data.csv',
        delimiter=',', encoding='latin1'
    )
    print(f"Data loaded: {data.shape[0]} rows, {data.shape[1]} columns")