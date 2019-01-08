#!/usr/bin/python3
from statistics import mean
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

filtered_samples = []

# 1% of samples take as outliers
p = len(samples)*0.01

sample_dict = {}
samples_range = [i for i in range(0, len(samples))]
old_value = -1

for i, s in enumerate(samples):
    distances, indices = nbrs.kneighbors([s])
    sample_dict[i] = mean(distances[0])

for key in sorted(sample_dict, key=sample_dict.get, reverse=True):
    value = sample_dict[key]
    if p <= 0 and old_value != value:
        break
    p -= 1
    samples_range.remove(key)

for i in samples_range:
    filtered_samples.append(samples[i])

# CLUSTER
cluster_nr = 10
classes = len(filtered_samples)*[0]
kmeans = KMeans(n_clusters=cluster_nr, random_state=0).fit(filtered_samples)
clusters = [[] for i in range(cluster_nr)]
# print cluster nr
for j, i in enumerate(filtered_samples):
    decode = [
        encoder_dict['gender'].inverse_transform([i[0]])[0],
        encoder_dict['race'].inverse_transform([i[1]])[0],
        encoder_dict['parents_edu'].inverse_transform([i[2]])[0],
        encoder_dict['lunch'].inverse_transform([i[3]])[0],
        encoder_dict['prep_test'].inverse_transform([i[4]])[0]
    ]
    decode.extend(i[5:8])
    cls = kmeans.predict([i])[0]
    clusters[cls].append(decode)
    classes[j] = cls
print(clusters)

# CLASS

X_train, X_test, y_train, y_test = train_test_split(filtered_samples, classes, test_size=0.33, random_state=42)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
# print classification report
print(classification_report(y_test, y_pred))
