import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import Imputer
data=pd.read_csv('train.csv')

#removing colume which has more then 1000 null values
data=data.dropna(thresh=len(data)-460, axis=1)


'''#plotting for finding Outliers
char_cols = data.dtypes.pipe(lambda x: x[x == 'int64']).index
for old_feature in char_cols:
    sns.boxplot(x=data[old_feature])
    plt.show()'''

#removing Outliers with z_score
from scipy import stats
char_cols = data.dtypes.pipe(lambda x: x[x == 'int64']).index
for old_feature in char_cols:
    z = np.abs(stats.zscore(data[old_feature]))
    new=np.where(z>10)
    for x in new:
        data=data.drop(x)

#breaking data into dependent and independent var
x_train=data.iloc[:,1:79]
y_train=data.iloc[:,-1:]


#filling missing values of all integer and float colume 
x_train=x_train.fillna(x_train.mean())
#filling missing value of categorical values with mode
char_cols = x_train.dtypes.pipe(lambda x: x[x == 'object']).index
for old_feature in char_cols:
    x_train[old_feature].fillna(x_train[old_feature].mode()[0], inplace=True)
    

char_cols = x_train.dtypes.pipe(lambda x: x[x == 'object']).index
for old_feature in char_cols:
    x_train = pd.get_dummies(x_train, columns=[old_feature], drop_first=True)
    

''' # Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
sc_y = StandardScaler()
y_train= sc_y.fit_transform(y_train)'''

#pricipal component analaysis
'''from sklearn.decomposition import PCA
pca=PCA(n_components=65)
x_train=pca.fit_transform(x_train)
variance=pca.explained_variance_ratio_'''
#train_test split
from sklearn.model_selection import train_test_split
x_train1,x_test1,y_train1,y_test1=train_test_split(x_train,y_train,test_size=0.2,random_state=0)

#model creation
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train1,y_train1)
y_pred=regressor.predict(x_test1)






from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import explained_variance_score as evs
from sklearn.metrics import r2_score
mse=mse(y_test1,y_pred)
evs=evs(y_test1,y_pred)
r2=r2_score(y_test1,y_pred)


#more advance Evloution K flod cross validation  #model for different dataset 
from sklearn.model_selection import cross_val_score as cvs
accuracies=cvs(estimator=regressor,X=x_train1,y=y_train1,cv=10)
accuracies.std()
accuracies.mean() 






