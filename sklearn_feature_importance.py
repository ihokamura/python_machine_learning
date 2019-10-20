"""
show feature importance of random forest
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split

from data_utility import WineData


def main():
    # prepare sample data and target variable
    wine_data = WineData()
    X = wine_data.X
    y = wine_data.y

    # split sample data into training data and test data and standardize them
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    # fit classifiers
    classifier = RandomForestClassifier(n_estimators=500, random_state=1).fit(X_train, y_train)

    # show score
    print('test accuracy:', classifier.score(X_test, y_test))

    # show feature importances
    importances = classifier.feature_importances_
    indices = np.argsort(importances)[::-1]
    features = np.array(wine_data.features)

    print('[importance feature (all)]')
    for rank, index in enumerate(indices, start=1):
        print('{rank:2d}) {feature:30s} {importance:f}'.format(
            rank=rank,
            feature=features[index],
            importance=importances[index]))

    plt.title('feature importance')
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), features[indices], rotation=90)
    plt.tight_layout()
    plt.show()

    # select features from model
    select = SelectFromModel(classifier, threshold=0.1, prefit=True)
    X_selected = select.transform(X_train)
    print('[importance feature (> 0.1)]')
    for i in range(X_selected.shape[1]):
        print('{rank:2d}) {feature:30s} {importance:f}'.format(
            rank=i + 1,
            feature=features[indices[i]],
            importance=importances[indices[i]]))


if __name__ == '__main__':
    main()
