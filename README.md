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
      import pandas as pd
     df=pd.read_csv("/content/Encoding Data.csv")
     df


  ![Screenshot 2025-05-11 114410](https://github.com/user-attachments/assets/e805836c-40c8-4aae-bc48-f756e016d4c1)



  ~~~
    from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
    pm=['Hot','Warm','Cold']
    e1=OrdinalEncoder(categories=[pm])
    e1.fit_transform(df[["ord_2"]])
~~~
 ![Screenshot 2025-05-11 114419](https://github.com/user-attachments/assets/afd232c6-0cf4-4202-8352-755f4f6ce0b3)




~~~
    df['bo2']=e1.fit_transform(df[["ord_2"]])
    df
~~~

 ![Screenshot 2025-05-11 114428](https://github.com/user-attachments/assets/9bb4f0fe-1d6d-45b2-a06d-8e594ee697d8)


~~~
    df['bo2']=e1.fit_transform(df[["ord_2"]])
    df
~~~
  ![Screenshot 2025-05-11 114438](https://github.com/user-attachments/assets/20feb9d2-e813-45c3-8886-3dc982779906)


~~~
    le=LabelEncoder()
    dfc=df.copy()
    dfc['ord_2']=le.fit_transform(dfc['ord_2'])
    dfc
~~~
  ![Screenshot 2025-05-11 114456](https://github.com/user-attachments/assets/cdefe1af-ab06-49f7-8ba4-a17fc61391c9)

  ~~~
    from sklearn.preprocessing import OneHotEncoder
    ohe=OneHotEncoder(sparse=False)
    df2=df.copy()
    enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
    df2=pd.concat([df2,enc],axis=1)
    df2
~~~

  ![Screenshot 2025-05-11 114505](https://github.com/user-attachments/assets/589f3116-a3e4-481e-90e6-105618bd9155)


  ~~~
    pd.get_dummies(df2,columns=["nom_0"])
~~~

 ![Screenshot 2025-05-11 114513](https://github.com/user-attachments/assets/6d4b3246-fd50-4e50-aee1-30f5646cb67e)


  ~~~
    pip install --upgrade category_encoders
  ~~~

  ![Screenshot 2025-05-11 114513](https://github.com/user-attachments/assets/88407b0a-8f82-4dac-b535-09df6d85d8b9)


~~~
    from category_encoders import BinaryEncoder
    df=pd.read_csv("/content/data.csv")
    be=BinaryEncoder()
    nd=be.fit_transform(df['Ord_2'])
    fb=pd.concat([df,nd],axis=1)
    dfb1=df.copy()
    dfb
 
 ~~~

 ![Screenshot 2025-05-11 114522](https://github.com/user-attachments/assets/9e4cf070-d6e1-45be-8d68-bd1149779201)


 ~~~
    from category_encoders import TargetEncoder
    te=TargetEncoder()
    cc=df.copy()
    new=te.fit_transform(X=cc["City"],y=cc["Target"])
    cc=pd.concat([cc,new],axis=1)
    cc
~~~

 ![Screenshot 2025-05-11 114530](https://github.com/user-attachments/assets/526744a6-5832-4555-953a-54fd91ccb8e3)


~~~
    import pandas as pd
    from scipy import stats
    import numpy as np
    df=pd.read_csv("/content/Data_to_Transform.csv")
    df
~~~

  ![Screenshot 2025-05-11 114541](https://github.com/user-attachments/assets/0f9cd149-cd6e-4687-a123-874f7ca84f9e)

~~~
    df.skew()
~~~

 ![Screenshot 2025-05-11 114552](https://github.com/user-attachments/assets/d27b4a88-4058-47ad-95f6-cef964347612)


~~~
    np.log(df["Highly Positive Skew"])
~~~

 ![Screenshot 2025-05-11 114604](https://github.com/user-attachments/assets/92fcaf24-9006-4e7a-b75c-e3071953c96e)


~~~
    np.reciprocal(df["Moderate Positive Skew"])
~~~

  ![Screenshot 2025-05-11 114612](https://github.com/user-attachments/assets/912b478f-cdf3-48be-a8da-65c280ce5402)


~~~
    np.sqrt(df["Highly Positive Skew"])
~~~

  ![Screenshot 2025-05-11 114619](https://github.com/user-attachments/assets/072d2a54-d3aa-4db1-aabe-6a8c7786ded8)

~~~
    np.square(df["Highly Positive Skew"])
~~~

  ![Screenshot 2025-05-11 114625](https://github.com/user-attachments/assets/cf5e5dc3-5ca5-4672-b4b9-21a8ac994f4a)


~~~
   df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
   df
~~~

  ![Screenshot 2025-05-11 114633](https://github.com/user-attachments/assets/2e514398-4360-4a17-9762-49dacbc7402e)


~~~
    df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
    df.skew()
~~~

  ![Screenshot 2025-05-11 114640](https://github.com/user-attachments/assets/d4b5f6d5-4493-434b-8ecb-fd07f43730cc)


~~~
    df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
    df.skew()

~~~
  ![Screenshot 2025-05-11 114646](https://github.com/user-attachments/assets/7c9163c6-c715-4725-a77a-7b033d78b379)


~~~
   import matplotlib.pyplot as plt
   import seaborn as sns
   import statsmodels.api as sm
   import scipy.stats as stats

   sm.qqplot(df["Moderate Negative Skew"],line='45')

   plt.show()
  ~~~

  ![Screenshot 2025-05-11 114653](https://github.com/user-attachments/assets/d9abc742-a282-4160-99c3-ffa1f07bd23a)


~~~
    sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
~~~

  ![Screenshot 2025-05-11 114702](https://github.com/user-attachments/assets/07f0178a-5fc5-48e4-8585-60686603431a)


~~~
    from sklearn.preprocessing import QuantileTransformer
    qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

    df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

    sm.qqplot(df["Moderate Negative Skew"],line='45')
    plt.show()
    
~~~

  ![Screenshot 2025-05-11 114710](https://github.com/user-attachments/assets/bc3d7f6f-f35b-46d1-a375-26e0a3077ebc)

~~~
# RESULT:
      Hence performing Feature Encoding and Transformation process is Successful.

       
