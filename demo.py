from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

path="G:\Mini project reference\StressLevelDataset.csv.xls"

data = pd.read_csv(path)
data.head()

X = data.drop("stress_level",axis=1)
y = data["stress_level"]
print(y)