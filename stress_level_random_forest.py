from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

path="C:/Users/ashok/OneDrive/Desktop/Mini project reference/StressLevelDataset.csv.xls"

data = pd.read_csv(path)
data.head()

X = data.drop("stress_level",axis=1)
y = data["stress_level"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf_classifier = RandomForestClassifier(n_estimators=100,random_state=42)

rf_classifier.fit(X_train_scaled,y_train)

y_pred = rf_classifier.predict(X_test_scaled)

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
result = confusion_matrix(y_test, y_pred)

'''sns.heatmap(result,
            annot=True,
            fmt='g',
            xticklabels=[0,1,2],
            yticklabels=[0,1,2])
plt.ylabel('Prediction',fontsize=13)
plt.xlabel('Actual',fontsize=13)
plt.title('Confusion Matrix',fontsize=17)
plt.show()'''

result1 = classification_report(y_test,y_pred)
print("Classification Report:")
print(result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)

feature_importances = rf_classifier.feature_importances_
importance_df = pd.DataFrame({'Feature':X.columns,'Importance':feature_importances})
importance_df = importance_df.sort_values(by='Importance',ascending=False)
folder_path='C:/Users/ashok/OneDrive/Desktop/Mini project reference'
importance_df.to_csv(folder_path+'feature_importance.csv', index=False)
print("Predicted Stress Levels:")
for stress_level in y_pred:
    print(stress_level)


import joblib
joblib.dump(rf_classifier,'C:/Users/ashok/OneDrive/Desktop/Mini project reference/random_forest_model.pkl')
joblib.dump(scaler,'C:/Users/ashok/OneDrive/Desktop/Mini project reference/scaler.pkl')

