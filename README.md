# Asg4
ai in healthcare assignment 4 implementing GradientBoostClassifier for hd_data.csv

Classification accuracy: 0.6172839506172839
Classification recall: 0.53125
Classification precision: 0.5151515151515151
Classification f1_score: 0.5230769230769231

with parameters: 
clf = GradientBoostingClassifier(n_estimators=10000, learning_rate=0.1, max_depth=10, random_state=0).fit(X_train_c, y_train_c)
