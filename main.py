#!/usr/bin/python3
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

data = pd.read_csv('StudentsPerformance.csv')
samples = data.values


# import ed_stat
# ed_stat.race_percentage(samples)

# OUTLIERS
edu_dict = {"bachelor's degree": 0.8, 'some college': 0.6, "master's degree": 1,
            "associate's degree": 0.8, 'high school': 0.5, 'some high school': 0}

encoder_dict = {
    'gender': LabelEncoder().fit(samples[:, 0]), 'race': LabelEncoder().fit(samples[:, 1]),
    'parents_edu': LabelEncoder().fit(samples[:, 2]), 'lunch': LabelEncoder().fit(samples[:, 3]),
    'prep_test': LabelEncoder().fit(samples[:, 4])
}


def gower_distance(e1, e2):
    result = 0
    # categorical
    for i in [0, 1, 3, 4]:
        if e1[i] != e2[i]:
            result += 1

    # exam scores
    for i in [5, 6, 7]:

    result += abs(edu_dict[encoder_dict['parents_edu'].inverse_transform([int(e1[2])])[0]] -
                  edu_dict[encoder_dict['parents_edu'].inverse_transform([int(e2[2])])[0]])
    return result


samples[:, 0] = encoder_dict['gender'].transform(samples[:, 0])
samples[:, 1] = encoder_dict['race'].transform(samples[:, 1])
samples[:, 2] = encoder_dict['parents_edu'].transform(samples[:, 2])
samples[:, 3] = encoder_dict['lunch'].transform(samples[:, 3])
samples[:, 4] = encoder_dict['prep_test'].transform(samples[:, 4])


print(encoder_dict['parents_edu'].inverse_transform([1]))

nbrs = NearestNeighbors(n_neighbors=5, algorithm='brute',
                        metric=gower_distance).fit(samples)

knn = nbrs.kneighbors([samples[0]])
print(knn[0])
