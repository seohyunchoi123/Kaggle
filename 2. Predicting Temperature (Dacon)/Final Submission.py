import pandas as pd

result1 = pd.read_csv("result1.csv")
result2 = pd.read_csv("result2.csv")

y18 = result1['Y18'] * 0.9 + result1['Y18'] * 0.1
result1['Y18'] = y18
result1.to_csv("submission.csv", index=False)