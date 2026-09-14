import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_heart_data(filepath='Heart.csv',
                   standardize=True,
                   test_size=0.2,
                   random_state=42):
    df = pd.read_csv(filepath)

    if 'id' in df.columns:
        df = df.drop('id', axis=1)

    le = LabelEncoder()
    df['famhist_encoded'] = le.fit_transform(df['famhist'])
    df = df.drop('famhist', axis=1)

    X = df.drop('chd', axis=1).values
    y = df['chd'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    feature_names = ['sbp', 'tobacco', 'ldl', 'adiposity', 'typea',
                    'obesity', 'alcohol', 'age', 'famhist']
    feature_info = pd.DataFrame({
        'feature': feature_names,
        'type': ['continuous'] * 8 + ['categorical'],
        'description': [
            'Systolic blood pressure',
            'Cumulative tobacco (kg)',
            'LDL cholesterol',
            'Adiposity index',
            'Type-A behavior score',
            'Obesity index',
            'Alcohol consumption',
            'Age (years)',
            'Family history (1=Present, 0=Absent)'
        ]
    })

    return X_train, X_test, y_train, y_test, feature_info


def load_fev_data(filepath='fev.csv',
                 standardize=True,
                 test_size=0.2,
                 random_state=42):
    df = pd.read_csv(filepath)

    if 'id' in df.columns:
        df = df.drop('id', axis=1)

    le_sex = LabelEncoder()
    le_smoke = LabelEncoder()

    df['sex_encoded'] = le_sex.fit_transform(df['sex'])
    df['smoke_encoded'] = le_smoke.fit_transform(df['smoke'])

    df = df.drop(['sex', 'smoke'], axis=1)

    X = df.drop('fev', axis=1).values
    y = df['fev'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if standardize:
        scaler = StandardScaler()
        X_train_cont = X_train[:, :2]
        X_test_cont = X_test[:, :2]

        X_train_cont = scaler.fit_transform(X_train_cont)
        X_test_cont = scaler.transform(X_test_cont)

        X_train[:, :2] = X_train_cont
        X_test[:, :2] = X_test_cont

    feature_names = ['age', 'height', 'sex', 'smoke']
    feature_info = pd.DataFrame({
        'feature': feature_names,
        'type': ['continuous', 'continuous', 'categorical', 'categorical'],
        'description': [
            'Age in years',
            'Height in inches',
            'Gender (1=male, 0=female)',
            'Smoking status (1=non-current, 0=current)'
        ]
    })

    return X_train, X_test, y_train, y_test, feature_info


def get_heart_data_summary(filepath='Heart.csv'):
    df = pd.read_csv(filepath)
    if 'id' in df.columns:
        df = df.drop('id', axis=1)

    summary = df.describe()
    summary.loc['missing'] = df.isnull().sum()

    chd_counts = df['chd'].value_counts()
    summary.loc['CHD=0 (healthy)'] = chd_counts.get(0, 0)
    summary.loc['CHD=1 (disease)'] = chd_counts.get(1, 0)

    return summary


def get_fev_data_summary(filepath='fev.csv'):
    df = pd.read_csv(filepath)
    if 'id' in df.columns:
        df = df.drop('id', axis=1)

    summary = df.describe()
    summary.loc['missing'] = df.isnull().sum()

    sex_counts = df['sex'].value_counts()
    smoke_counts = df['smoke'].value_counts()

    summary.loc['sex: female'] = sex_counts.get('female', 0)
    summary.loc['sex: male'] = sex_counts.get('male', 0)
    summary.loc['smoke: current'] = smoke_counts.get('current smoker', 0)
    summary.loc['smoke: non-current'] = smoke_counts.get('non-current smoker', 0)

    return summary
