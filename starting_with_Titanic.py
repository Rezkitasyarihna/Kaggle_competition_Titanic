# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %%
df = pd.read_csv('/train.csv')
df.head()

# %%
#we wanna know survival percentage of each category

#Pclass
pclass_1 = df.loc[df.Pclass == 1]["Survived"]
rate_pclass_1 = sum(pclass_1)/len(pclass_1)
print("% of people from class #1 who survived:", rate_pclass_1)
pclass_2 = df.loc[df.Pclass == 2]["Survived"]
rate_pclass_2 = sum(pclass_2)/len(pclass_2)
print("% of people from class #2 who survived:", rate_pclass_2)
pclass_3 = df.loc[df.Pclass == 3]["Survived"]
rate_pclass_3 = sum(pclass_3)/len(pclass_3)
print("% of people from class #3 who survived:", rate_pclass_3)

#Sex
women = df.loc[df.Sex == "female"]["Survived"]
rate_women = sum(women)/len(women)
print("% of women who survived:", rate_women)
men = df.loc[df.Sex == "male"]["Survived"]
rate_men = sum(men)/len(men)
print("% of men who survived:", rate_men)


# %% [markdown]
# That information gave us a brief visualisation about survive probability for train set based on class and gender. We can make a better prediction by adding 'Parch' and 'SibSp' variable into consideration. 

# %%
test = pd.read_csv('/kaggle/input/titanic/test.csv')
from sklearn.ensemble import RandomForestClassifier

y = df["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(df[features])
X_test = pd.get_dummies(test[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
output.to_csv('prediction_result.csv', index=False)
print("Your submission was successfully saved!")

# %%
df_prediction_result = pd.read_csv('/kaggle/working/prediction_result.csv')
df_prediction_result.groupby('Survived').size()

# %%