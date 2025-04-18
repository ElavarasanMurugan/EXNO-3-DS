## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df = pd.read_csv("/content/Encoding Data.csv")
df
```
![img-1](https://github.com/user-attachments/assets/444da500-3f9c-469f-ab58-a0ca222e62e7)

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm = ['Hot','Warm','Cold']
e1 = OrdinalEncoder(categories=[pm])
e1.fit_transform(df[['ord_2']])
```
![img-2](https://github.com/user-attachments/assets/da1cdfa8-6a3c-4a30-91a8-4a66a21f5c5b)

```
df['bo2'] = e1.fit_transform(df[["ord_2"]])
df
```
![img-3](https://github.com/user-attachments/assets/261c71f6-db13-468e-ad23-326f75fbe3a3)

```
le = LabelEncoder()
dfc = df.copy()
dfc['ord_2'] = le.fit_transform(dfc['ord_2'])
dfc
```
![img-4](https://github.com/user-attachments/assets/4230bbab-3948-4a7a-966d-52fdef247f09)

```
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
df2 = df.copy()
enc = pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2 = pd.concat([df2,enc],axis=1)
df2
```
![img-5](https://github.com/user-attachments/assets/546c2086-58b0-4029-9bdb-a5dca0e30e89)

```
pd.get_dummies(df2,columns=["nom_0"])
```
![img-6](https://github.com/user-attachments/assets/57ecd59d-29ca-43ae-8eb5-27b66b68a655)

```
pip install --upgrade category_encoders
```
![img-7](https://github.com/user-attachments/assets/74b0ffb8-000f-41c4-844e-d8b28ace2f8c)

```
from category_encoders import BinaryEncoder
df = pd.read_csv("/content/data.csv")
df
```

![img-8](https://github.com/user-attachments/assets/1f5302aa-1c25-46ea-92d1-d2b07f190c80)

```
be = BinaryEncoder()
nd = be.fit_transform(df['Ord_2'])
dfb = pd.concat([df,nd],axis=1)
dfb1 = df.copy()
dfb
```

![img-9](https://github.com/user-attachments/assets/b183107d-66d4-4f36-83fd-ff9e43bf1e38)

```
from category_encoders import TargetEncoder
te = TargetEncoder()
cc = df.copy()
new = te.fit_transform(X=cc["City"],y=cc["Target"])
cc = pd.concat([cc,new],axis=1)
cc
```

![img-10](https://github.com/user-attachments/assets/e56ad75e-c78d-4fa5-aed9-1d3475a1f5d0)

Feature Transformation
```
import pandas as pd
import numpy as np
from scipy import stats
df = pd.read_csv("/content/Data_to_Transform.csv")
df
```

![img-11](https://github.com/user-attachments/assets/52e80a28-ac60-4509-9772-f3bba4427256)

```
df.skew()
```

![img-12](https://github.com/user-attachments/assets/8f181956-0cd8-4517-b079-1ddc6c4ffea5)

```
np.log(df["Highly Positive Skew"])
```

![img-13](https://github.com/user-attachments/assets/9eb9e2b6-65fe-43e7-8efe-fe3c7ef87500)

```
np.reciprocal(df["Moderate Positive Skew"])
```

![img-14](https://github.com/user-attachments/assets/0328d97b-3e44-47f7-8dc8-91c4af3fd726)

```
np.sqrt(df["Highly Positive Skew"])
```

![img-15](https://github.com/user-attachments/assets/53988917-bdd5-4547-860b-76e4de5e96d5)

```
np.square(df["Highly Positive Skew"])
```

![img-16](https://github.com/user-attachments/assets/2d061ab2-cbe2-4856-87f6-94fb7a9fbd5c)

```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```

![img-17](https://github.com/user-attachments/assets/b1e3d361-55d2-4e02-a7d1-415cfeb5b90e)

```
df["Moderate Negative Skew_yeojohnson"],parameters = stats.yeojohnson(df["Moderate Negative Skew"])
df
```

![img-18](https://github.com/user-attachments/assets/57d05e02-6b50-44b2-a7b4-22a78277333c)

```
df.skew()
```

![img-19](https://github.com/user-attachments/assets/6fbb626b-144f-4602-ae6d-d6d257bdfa16)

```
df["Highly Negative Skew_yeojohnson"],parameters = stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```

![img-20](https://github.com/user-attachments/assets/81a2e4e7-6613-45a9-84d4-d553d7628634)

```
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"] = qt.fit_transform(df[["Moderate Negative Skew"]])
df
```

![img-21](https://github.com/user-attachments/assets/db525c0f-93f1-4e2a-80fc-aa7e972e4c8c)

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![img-22](https://github.com/user-attachments/assets/c8ca0986-e839-46fc-98ac-9d73f925c476)

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

![img-23](https://github.com/user-attachments/assets/15b0475b-70f8-4d56-a402-5e1ce235e205)

```
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew"] = qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

![img-24](https://github.com/user-attachments/assets/1798f14e-2184-4d6a-bbc9-7dae79b94c1a)

```
df["Highly Negative Skew_1"] = qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```

![img-25](https://github.com/user-attachments/assets/e7727549-2a04-4c26-835a-8180bceb6a26)

```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```

![img-26](https://github.com/user-attachments/assets/de656364-dcc7-4ab9-b4f8-bc4b36d88a0e)

# RESULT:
   Thus, the given data was successfully read, feature encoding and transformation were performed, and the resulting data was saved to a file.

       
