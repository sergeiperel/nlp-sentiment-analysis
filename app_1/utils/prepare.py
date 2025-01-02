import numpy as np
import pandas as pd
from scipy.sparse import hstack


def open_dataset(file_name):
    return pd.read_csv(file_name)

def prepare_dataset(df, val_size = 0.2):
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')
    df[['user_location', 'user_description', 'emojis']] = df[['user_location', 'user_description', 'emojis']].fillna(
        'None')
    df_encoded = pd.get_dummies(df, columns=['user_location', 'user_description'], drop_first=False)
    train_cutoff = df_encoded['date'].quantile(1-val_size)  # Дата, отсекающая 60% наблюдений
    train_df = df_encoded[df_encoded['date'] <= train_cutoff]
    val_df = df_encoded[df_encoded['date'] > train_cutoff]
    X_train = train_df.drop(columns=['feeling', 'date'])
    y_train = train_df['feeling']
    X_val = val_df.drop(columns=['feeling', 'date'])
    y_val = val_df['feeling']
    return X_train, y_train, X_val, y_val

def prepare_dataset_2(X_train, X_val, vectorizer, scaler):
    X_train_lemmatized = vectorizer.fit_transform(X_train['lemmatized_str'])
    X_val_lemmatized = vectorizer.transform(X_val['lemmatized_str'])
    numerical_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    X_train_numerical = scaler.fit_transform(X_train[numerical_features])
    X_val_numerical = scaler.transform(X_val[numerical_features])
    X_train_combined = hstack([X_train_lemmatized, X_train_numerical])
    X_val_combined = hstack([X_val_lemmatized, X_val_numerical])
    return X_train_combined, X_val_combined


def prepare_dataset_3(X_test: pd.DataFrame, vectorizer, scaler):
    if 'feeling' in X_test.columns:
        X_test = X_test.drop('feeling', axis=1)
    X_test_lemmatized = vectorizer.transform(X_test['lemmatized_str'])
    numerical_features = X_test.select_dtypes(include=[np.number]).columns.tolist()
    X_test_numerical = scaler.transform(X_test[numerical_features])
    X_test_combined = hstack([X_test_lemmatized, X_test_numerical])
    return X_test_combined



