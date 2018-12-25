from sklearn.ensemble import IsolationForest

edu_dict = {"bachelor's degree": 0.8, 'some college': 0.6, "master's degree": 1,
            "associate's degree": 0.8, 'high school': 0.5, 'some high school': 0}


def gower_distance(e1, e2):
    result = 0
    # categorical
    for i in [0, 1, 3, 4]:
        if(e1[i] == e2[i]):
            result += 1

    # exam scores
    for i in [5, 6, 7]:
        result += abs(e1[i] - e2[i])/100

    # parents education
    result += abs(edu_dict[e1[2]] - edu_dict[e2[2]])
    return result


def outliers(samples):
    clf = IsolationForest()
    clf.fit(samples)
    return clf.fit_predict(samples)
