import pandas as pd
import os.path as op
import preprocessing as pr
import training as tr


def read_data():
    df = 1
    return df


def data_processing(df):
    # 데이터 전처리

    return df


if __name__ == "__main__":
    df = read_data()
    df = data_processing(df)

    # 데이터 전처리
    pr.data_analysis(df)

    # 데이터 훈련
    tr.prediction(df)
