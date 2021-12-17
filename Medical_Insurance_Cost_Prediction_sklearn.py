import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from google.colab import drive

drive.mount("/content/drive")

medi = pd.read_csv("drive/kongkea/Dataset/insurance.csv")
medi.head(5)
medi.shape
medi.describe()
medi.isnull().sum()
sns.set_theme(style="whitegrid")
sns.boxplot(medi["charges"])
sns.boxplot(medi["bmi"])
medi[["sex", "age"]].groupby("sex").agg(["mean", "count"])
medi["smoker"].value_counts()
medi[["sex", "children"]].groupby("sex").agg(["mean"])
plt.figure(figsize=(8, 6))
sns.scatterplot(data=medi, x="age", y="bmi", hue="sex", style="sex")
sns.scatterplot(data=medi, x="age", y="charges", hue="sex", style="sex")
sns.scatterplot(data=medi, x="age", y="charges", hue="smoker", style="smoker")
medi = medi[medi["bmi"] < 47]
medi.shape
medi["smoker"].value_counts()
medi["region"].value_counts()
medi["sex"].value_counts()


def label_encoded(feat):
    le = LabelEncoder()
    le.fit(feat)
    print(feat.name, le.classes_)
    return le.transform(feat)


name_list = ["sex", "smoker", "region"]
for name in name_list:
    medi[name] = label_encoded(medi[name])
medi.head(3)
plt.figure(figsize=(7, 7))
sns.heatmap(medi.corr(), annot=True, cmap="viridis", linewidths=0.5)
y = medi["charges"]
X = medi.drop(["charges"], axis=1)

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42)
random_model = RandomForestRegressor(n_estimators=250, n_jobs=-1)
random_model.fit(Xtrain, ytrain)
y_pred = random_model.predict(Xtest)
random_model_accuracy = round(random_model.score(Xtrain, ytrain) * 100, 2)
print(round(random_model_accuracy, 2), "%")
random_model_accuracy1 = round(random_model.score(Xtest, ytest) * 100, 2)
print(round(random_model_accuracy1, 2), "%")

saved_model = pickle.dump(
    random_model, open("drive/kongkea/Dataset/Models/Medical.pickle", "wb")
)
