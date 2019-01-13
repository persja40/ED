#!/usr/bin/python3
from statistics import mean
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('StudentsPerformance.csv')
samples = data.values

# import ed_stat
# ed_stat.race_percentage(samples)

n_neighbors = 5

# OUTLIERS
edu_dict = {"bachelor's degree": 0.8, 'some college': 0.6, "master's degree": 1,
            "associate's degree": 0.8, 'high school': 0.5, 'some high school': 0}

encoder_dict = {
    'gender': LabelEncoder().fit(samples[:, 0]), 'race': LabelEncoder().fit(samples[:, 1]),
    'parents_edu': LabelEncoder().fit(samples[:, 2]), 'lunch': LabelEncoder().fit(samples[:, 3]),
    'prep_test': LabelEncoder().fit(samples[:, 4])
}

exams_list = ["math score", "reading score", "writing score"]


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

nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute',
                        metric=gower_distance).fit(samples)

# 1% of samples take as outliers
p = len(samples) * 0.01

sample_dict = {}
samples_range = [i for i in range(len(samples))]
outliers = []
old_value = -1

for i, s in enumerate(samples):
    distances, indices = nbrs.kneighbors([s])
    sample_dict[i] = mean(distances[0])

for key in sorted(sample_dict, key=sample_dict.get, reverse=True):
    value = sample_dict[key]
    if p <= 0 and old_value != value:
        break
    old_value = value
    p -= 1
    outliers.append([key, value])
    samples_range.remove(key)

filtered_samples = [samples[i] for i in samples_range]

with open('stats/outliers.dat', 'w') as f:
    for o in outliers:
        i = o[0]
        decode = [
            encoder_dict['gender'].inverse_transform([samples[i][0]])[0],
            encoder_dict['race'].inverse_transform([samples[i][1]])[0],
            encoder_dict['parents_edu'].inverse_transform([samples[i][2]])[0],
            encoder_dict['lunch'].inverse_transform([samples[i][3]])[0],
            encoder_dict['prep_test'].inverse_transform([samples[i][4]])[0]
        ]
        decode.extend(samples[i][5:8])

        f.write(str(i) + ' \tdistance={:.2f}'.format(o[1]) + ', ' + str(decode) + '\n')

# CLUSTER
cluster_nr = 10
classes = len(filtered_samples) * [0]
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

knc = KNeighborsClassifier(n_neighbors=n_neighbors - 1).fit(X_train, y_train)
y_pred = knc.predict(X_test)
# print classification report
target_names = ['class ' + str(i) for i in range(cluster_nr)]
print(classification_report(y_test, y_pred, target_names=target_names))

class_stats = []

for i in range(cluster_nr):
    sts = []
    with open('stats/class ' + str(i) + '.dat', 'w') as f:
        for length in clusters[i]:
            f.write(str(length) + '\n')

        for key in encoder_dict.keys():
            f.write(key + '\n\t')
            for k in encoder_dict[key].classes_:
                v = sum(c.count(k) for c in clusters[i])
                sts.append([k, v])
                f.write(k + "=" + str(v) + "\t")
            f.write('\n')

        for j in range(len(exams_list)):
            v_min = min(c[j + 5] for c in clusters[i])
            v_max = max(c[j + 5] for c in clusters[i])
            v_avg = mean(c[j + 5] for c in clusters[i])
            sts.append([exams_list[j], v_min, v_max, v_avg])
            f.write(exams_list[j] + '\n\t')
            f.write("min={}, max={}, avg={}".format(v_min, v_max, v_avg))
            f.write('\n')
    class_stats.append(sts)

print(len(class_stats[0]))

length = len(class_stats[0]) - 3

with open('stats/class_stats.dat', 'w') as f:
    for i in range(cluster_nr):
        f.write('class ' + str(i) + ' ')
        for j in range(len(class_stats[i]) - 3):
            f.write(str(class_stats[i][j][0]) + '=' + str(class_stats[i][j][1]) + '  \t')
        for j in range(3):
            f.write(str(class_stats[i][j + length][0]) + ':' + 'min={}, max={}, avg={}'.format(
                class_stats[i][j + length][1], class_stats[i][j + length][2], class_stats[i][j + length][3]) + '\t')
        f.write('\n')
