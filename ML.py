# ========================================================================================================
# Pokemon Battle
# ========================================================================================================
import pandas as pd

df = pd.read_csv(
    'dataGabungan.csv', 
    index_col = 0
)

dfX = df.drop(['id1', 'id2', 'winner'], axis = 1)
dfY = df['winner']

from sklearn.model_selection import train_test_split
xtr, xts, ytr, yts = train_test_split(
    dfX,
    dfY,
    test_size = 0.1
)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver='liblinear',multi_class='auto')
model.fit(xtr, ytr)

import joblib
joblib.dump(model,'ML.joblib')