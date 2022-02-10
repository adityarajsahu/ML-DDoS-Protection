import pandas as pd
from numpy import int64


def processor(df):
    df = df.drop_duplicates(keep='first')
    df = df[['src', 'dt', 'label']]
    df['src'] = df['src'].astype(str)
    df[['ip1', 'ip2', 'ip3', 'ip4']] = df['src'].str.split('.', expand=True)
    df = df.drop(columns='src', axis=1)
    df['ip1'] = df['ip1'].astype(int64)
    df['ip2'] = df['ip2'].astype(int64)
    df['ip3'] = df['ip3'].astype(int64)
    df['ip4'] = df['ip4'].astype(int64)
    return df
