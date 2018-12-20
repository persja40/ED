#!/usr/bin/python3
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('StudentsPerformance.csv')
samples = data.values

import ed_stat
ed_stat.race_percentage(samples)
