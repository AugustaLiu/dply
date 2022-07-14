import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression


df = pd.read_csv("churn_agg1.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1] # churn (1) or non-churn (0)

model = LogisticRegression(class_weight = 'balanced',
                                    penalty = 'l2',
                                    C = 0.9,
                                    solver = 'saga',
                                    #l1_ratio = 0.5,
                                    n_jobs = -1, 
                                    random_state = 2)
model.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
