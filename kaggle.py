import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import Imputer
data=pd.read_csv('train.csv')
data2=pd.read_csv('test.csv')
#removing colume which has more then 1000 null values
data=data.dropna(thresh=len(data)-460, axis=1)


#removing Outliers with z_score
from scipy import stats
char_cols = data.dtypes.pipe(lambda x: x[x == 'int64']).index
for old_feature in char_cols:
    z = np.abs(stats.zscore(data[old_feature]))
    new=np.where(z>15)
    for x in new:
        data=data.drop(x)
        
#breaking data into dependent and independent var
x_train=data.iloc[:,1:79]
y_train=data.iloc[:,-1:]
y_train=np.log(y_train.SalePrice)



#filling missing values of all integer and float colume 
x_train=x_train.fillna(x_train.mean())
data2=data2.fillna(data2.mean())

#filling missing value of categorical values with mode
char_cols = x_train.dtypes.pipe(lambda x: x[x == 'object']).index
for old_feature in char_cols:
    x_train[old_feature].fillna(x_train[old_feature].mode()[0], inplace=True)
    

char_cols = data2.dtypes.pipe(lambda x: x[x == 'object']).index
for old_feature in char_cols:
    data2[old_feature].fillna(data2[old_feature].mode()[0], inplace=True)
    

char_cols = x_train.dtypes.pipe(lambda x: x[x == 'object']).index
for old_feature in char_cols:
    x_train = pd.get_dummies(x_train, columns=[old_feature], drop_first=True)

char_cols = data2.dtypes.pipe(lambda x: x[x == 'object']).index
for old_feature in char_cols:
    data2 = pd.get_dummies(data2, columns=[old_feature], drop_first=True)

submission = pd.DataFrame()
submission['Id'] = data2.Id
data2=data2.iloc[:,1:79]

    
'''# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)

sc_y = StandardScaler()
y_train= sc_y.fit_transform(y_train)

sc_datax = StandardScaler()
data2 = sc_datax.fit_transform(data2)'''


#pricipal component analaysis
from sklearn.decomposition import PCA
pca=PCA(n_components=65)
x_train=pca.fit_transform(x_train)
variance=pca.explained_variance_ratio_

#pricipal component analaysis
from sklearn.decomposition import PCA
pca=PCA(n_components=65)
data2=pca.fit_transform(data2)
variance=pca.explained_variance_ratio_

#graph




#model creation
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(data2)





y_new_inverse = np.exp(y_pred)


submission['SalePrice'] = y_new_inverse

submission.to_csv('submission_new.csv', index=False)







