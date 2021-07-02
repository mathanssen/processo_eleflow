import pandas as pd
from collections import Counter
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def get_missing(df):
    """Retorna dataframe com percentual de missing por coluna"""

    missing = df.isnull().sum()
    missing_percentage = df.isnull().sum() / df.isnull().count() * 100
    missing_percentage = round(missing_percentage, 1)
    missing_data = pd.concat([missing, missing_percentage], axis=1, keys=["Total", "%"])
    missing_data = missing_data[missing_data["Total"] > 0].sort_values(
        by=["%"], ascending=False
    )

    return missing_data


def split_column(df, col, min_frequency=10):
    """
    Retorna o dataframe com as novas colunas e
    outro com os ratings para os diferentes elementos da coluna
    """

    # Criação de contadores para frequência e soma de rating
    c1, c2 = Counter(), Counter()
    df_counter = df.copy()
    df_counter = df_counter[df_counter[col].notna()]
    index = df.columns.get_loc(col) + 1
    for row in df_counter.itertuples():
        for i in row[index].split(", "):
            c1[i] += row[9]
            c2[i] += 1

    # Conversão dos contadores para Series e filtro dos elementos dada uma frequência
    s = pd.Series(c1).div(pd.Series(c2))
    s1 = pd.Series(c2)
    s3 = s1[s1.values > min_frequency].index.values

    # Criação de um dataframe com os diferentes elementos
    df_col_rating = pd.DataFrame({col: s.index, "rating": s.values})
    df_col_rating = df_col_rating[df_col_rating[col].isin(s3)]
    df_col_rating = df_col_rating.sort_values(by="rating", ascending=False)

    # Criação das novas colunas e iteração do dataframe
    df_col = pd.DataFrame(columns=s3.tolist())
    df = pd.concat([df, df_col])

    for i, row in df.iterrows():
        element_string = str(row[col])
        element_lst = [x.strip() for x in element_string.split(",")]
        for e in element_lst:
            if e in df.columns:
                df[e][i] = 1

    for c in df_col.columns:
        df[c] = df[c].fillna(0)
        new_name = col + "_" + str(c)
        df.rename(columns={c: new_name}, inplace=True)

    return df, df_col_rating


def get_rating_by_frequency(df, col, min_frequency=10):
    """Retorna o rating médio da frequência de cada elemento da coluna"""

    c1, c2 = Counter(), Counter()
    df_counter = df.copy()
    df_counter = df_counter[df_counter[col].notna()]
    index = df.columns.get_loc(col) + 1
    for row in df_counter.itertuples():
        for i in row[index].split(", "):
            c1[i] += row[9]
            c2[i] += 1

    s = pd.Series(c1).div(pd.Series(c2))
    s1 = pd.Series(c2)
    s3 = s1[s1.values > min_frequency].index.values

    df_frequency = pd.concat([s1, s], axis=1)
    df_frequency.rename(columns={0: "frequency", 1: "rating"}, inplace=True)
    df_frequency = df_frequency.sort_values(by=["frequency"], ascending=False)
    df_frequency = df_frequency.groupby(by=["frequency"]).mean()

    return df_frequency


def get_rating_by_element(df, col):
    """Retorna o rating médio de cada elemento"""

    df = df.groupby([col])["rating"].mean().reset_index()

    return df
