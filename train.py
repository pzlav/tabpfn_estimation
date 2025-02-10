import pandas as pd
import numpy as np

from model import Model


def add_time_features(df, date_col='pickup_date'):
    '''
    Add simple time features to the dataframe
    '''
    df[date_col] = pd.to_datetime(df[date_col])
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    df['hour'] = df[date_col].dt.hour
    max_date = pd.to_datetime('2022-10-01')
    df['months_to_max'] = ((max_date.year - df['year']) * 12 + (max_date.month - df['month'])) + 2
    df.drop(date_col, axis=1, inplace=True)  
    return df


def loss(real_rates, predicted_rates):
    return np.average(abs(predicted_rates / real_rates - 1.0)) * 100.0


def train_and_validate():
    train_df = pd.read_csv('dataset/train.csv')
    valid_df = pd.read_csv('dataset/validation.csv')
    train_df = add_time_features(train_df)
    train_df = train_df[train_df['year'] >= 2021] # use only 2021 and 2022 data
    valid_df = add_time_features(valid_df)
    X_train = train_df.drop('rate', axis=1)
    y_train = train_df['rate']
    X_valid = valid_df.drop('rate', axis=1)
    y_valid = valid_df['rate']

    model = Model()
    predicted_rates = model.fit_predict(X_train, y_train, X_valid)

    mape = loss(y_valid, predicted_rates)
    mape = np.round(mape, 2)
    return mape


def generate_final_solution():
    # combine train and validation to improve final predictions
    train_df = pd.read_csv('dataset/train.csv')
    test_df = pd.read_csv('dataset/test.csv')
    train_df = add_time_features(train_df)
    train_df = train_df[train_df['year'] >= 2021] # use only 2021 and 2022 data
    test_df = add_time_features(test_df)
    X_train = train_df.drop('rate', axis=1)
    y_train = train_df['rate']

    model = Model()
    predicted_rates = model.fit_predict(X_train, y_train, test_df)

    # generate and save test predictions
    df_test = pd.read_csv('dataset/test.csv')
    df_test['predicted_rate'] = predicted_rates
    df_test.to_csv('dataset/predicted.csv', index=False)


if __name__ == "__main__":
    mape = train_and_validate()
    print(f'Accuracy of validation is {mape}%')

    if mape < 9:  # try to reach 9% or less for validation
        generate_final_solution()
        print("'predicted.csv' is generated, please send it to us")
