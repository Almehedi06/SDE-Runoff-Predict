import pandas as pd

def preprocess_data(file_path, gauge_ids_of_interest, years):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])

    start_dates = df.groupby('gauge_id')['date'].min().reset_index()
    start_dates.columns = ['gauge_id', 'start_date']
    df = df.merge(start_dates, on='gauge_id')
    df['end_date'] = df['start_date'] + pd.DateOffset(years=years)
    df = df[(df['date'] >= df['start_date']) & (df['date'] < df['end_date'])]
    df = df.drop(columns=['start_date', 'end_date'])
    df = df[df['gauge_id'].isin(gauge_ids_of_interest)].sort_values(by=['gauge_id', 'date'])
    df['streamflow'] = df['streamflow'] / df['streamflow'].mean()
    return df
