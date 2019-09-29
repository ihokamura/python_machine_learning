"""
show sample of how to use Pandas library for preprocessing
"""

from io import StringIO

import numpy as np
import pandas as pd
from sklearn import impute
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler

from data_utility import WineData


def handle_missing_values():
    # prepare sample data
    csv_data = """
    A,B,C,D
    1.0,2.0,3.0,4.0
    5.0,6.0,,8.0
    10.0,11.0,12.0,
    """
    df = pd.read_csv(StringIO(csv_data))

    print('[CSV data]')
    print(df)

    print('[count missing values for each features]')
    print(df.isnull().sum())

    print('[drop samples which has a NaN]')
    print(df.dropna())
    print('[drop features which has a NaN]')
    print(df.dropna(axis=1))
    print('[drop samples of which all the features are NaN]')
    print(df.dropna(how='all'))
    print('[drop samples which has less than 4 NaN]')
    print(df.dropna(thresh=4))
    print('[drop samples which has a NaN in specified features]')
    print(df.dropna(subset=['C']))

    print('[impute(interpolate) by mean]')
    imp = impute.SimpleImputer(missing_values=np.NaN, strategy='mean').fit(df.values)
    print(imp.transform(df.values))


def handle_category_data():
    # prepare sample data
    df = pd.DataFrame(
        [
            ['green', 'M', 10.1, 'class1'],
            ['red', 'L', 13.5, 'class2'],
            ['blue', 'XL', 15.3, 'class1'],
        ]
    )
    df.columns = ['color', 'size', 'price', 'class_label']
    print(df)

    # encode ordinal feature
    size_mapping = {'XL':3, 'L':2, 'M':1}
    df['size'] = df['size'].map(size_mapping)
    print(df)

    # encode class labels
    class_mapping = {label:index for index, label in enumerate(np.unique(df['class_label']))}
    df['class_label'] = df['class_label'].map(class_mapping)
    print(df)
    y = LabelEncoder().fit_transform(df['class_label'].values)
    print(y)

    # encode nominal feature by one-hot encoding (after encoding to number)
    X = df[['color', 'size', 'price']].values
    X[:, 0] = LabelEncoder().fit_transform(X[:, 0])
    ohe = OneHotEncoder(categorical_features=[0]).fit_transform(X)
    print(ohe.toarray())

    # execute one-hot encoding with Pandas
    print(pd.get_dummies(df[['color', 'size', 'price']]))
    print(pd.get_dummies(df[['color', 'size', 'price']], drop_first=True))


def split_dataset():
    # load wine dataset
    wine_data = WineData()
    df = pd.DataFrame(data=wine_data.X, columns=wine_data.features)
    df.insert(loc=0, column='label', value=wine_data.y)
    print(df.head())

    # split dataset into training dataset and test dataset
    X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y)
    print(X_train[:5, :])
    print(X_test[:5, :])
    print(np.bincount(y))
    print(np.bincount(y_train))
    print(np.bincount(y_test))


def scale_feature():
    data = np.array([0, 1, 2, 3, 4, 5]).reshape(-1, 1)
    data_norm = MinMaxScaler().fit_transform(data).ravel()
    data_std = StandardScaler().fit_transform(data).ravel()
    print('normalized:', data_norm)
    print('standardized:', data_std)


if __name__ == '__main__':
    handle_missing_values()
    handle_category_data()
    split_dataset()
    scale_feature()
