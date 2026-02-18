# classification
# gradient boost since dataset is <10,000
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier

df = pd.read_csv("data/hd_data.csv")
# converting string labels into binary labels for classification as per python primer
df['Heart Disease Present'] = df['Heart Disease'].replace({'Absence': 0, 'Presence': 1})

X_class = df[['Age', 'BP', 'Cholesterol', 'Max HR']]
Y_class = df['Heart Disease Present']
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_class, Y_class, test_size=0.3, random_state=42)


clf = GradientBoostingClassifier(n_estimators=10000, learning_rate=0.1,
    max_depth=10, random_state=0).fit(X_train_c, y_train_c)
clf.score(X_test_c, y_test_c)

y_pred_class = clf.predict(X_test_c)
accuracy = accuracy_score(y_test_c, y_pred_class)
recall = recall_score(y_test_c, y_pred_class)
precision = precision_score(y_test_c, y_pred_class)
f1_score = f1_score(y_test_c,y_pred_class )

print("Classification accuracy:", accuracy)
print("Classification recall:", recall)
print("Classification precision:", precision)
print("Classification f1_score:", f1_score)