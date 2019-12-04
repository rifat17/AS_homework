#!/usr/bin/env python
# coding: utf-8

# # Due Dec 5, 11:59 PM
# Programming Assignment
# 15 points
# Shovon Bhowmik Nov 17 (Edited Nov 17)
# Problems that has to be solved using python or matlab:
# 
# 1. 
#  * Make an excel sheet of the datas given for the correlation example in class. 
#  * Calculate correlation coefficient and find the relationship between the datas. 
#  * Also compare the result with the defined function used by python or matlab for establishing correlation between datas. 
#  * Also visualize the graphical representation of the datas.
# 
# 
# 
# 2. 
#  * Collect any dataset with two variables x and y from any site. **The dataset must be atleast of 100 items.**
#  * Find the predicted y value based on x using **simple linear regression equation.** 
#  
#  Then find 
#  
#  1. SSR,
#  2. SSE 
#  and 
#  3. Coefficient of determination by calculating difference between the actual y value and the predicted y value. 
#  4. Finally show the regression line based on your result graphically and find the accuracy of your model's performance.
# 
# N. B:
# 
# 1) You all are requested to submit the problems in the following link:
# https://drive.google.com/drive/folders/1qreLOzd0iN3ye4JfZ9S_O9O5qzTMnD-N?usp=sharing
# Upload your zip file which will be named after your roll number and there will be two files inside the zip file. Both files will be in .py or .mat format
# 
# 2) If I find same code, the code that will come second or next will be not accepted and he or she will get no marks in the assignment.
# 
# Please knock me if there is any question regarding the assignment.
# Thank you.

# In[68]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# DATASET 
# > https://archive.ics.uci.edu/ml/datasets/student+performance

# In[69]:


df = pd.read_csv('student-mat.csv', delimiter=';')


# In[70]:


df.head()


# In[71]:


mydf = df.iloc[ : , -3:-1]


# In[72]:


# G1 = Grade1
# G2 = Grade2
# 
mydf.head()


# In[73]:


dataX = mydf.iloc[ : , 0:1]
dataY = mydf.iloc[ : , 1:]


# In[74]:


dataX.head()


# In[75]:


dataY.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# \begin{equation} r = ( sum((Xi - meanX) * (Yi - meanY) ) / sqrt( sum( squre( Xi - meanX) * squre(Yi - meanY) ) )) \end{equation}

# In[76]:


r = ( np.sum( (dataX.values - dataX.values.mean()) * (dataY.values - dataY.values.mean()) ) / np.sqrt( np.sum( np.square(dataX.values - dataX.values.mean()) ) * np.sum( np.square(dataY.values - dataY.values.mean()) ) ) )


# In[77]:


r


# In[78]:


meanX = dataX.values.mean()


# In[79]:


meanY = dataY.values.mean()


# In[80]:


sumOfMultiplyOfMeanXY = np.sum((dataX.values - meanX) * (dataY.values-meanY))


# In[81]:


meanX


# In[82]:


meanY


# In[83]:


sumOfMultiplyOfMeanXY


# In[84]:


np.sum( (dataX.values - dataX.values.mean()) * (dataY.values - dataY.values.mean()))


# In[85]:


np.sqrt( np.sum( np.square(dataX.values - dataX.values.mean()) ) * np.sum( np.square(dataY.values - dataY.values.mean()) ) )


# In[86]:


sumOfSqureOfX = np.sum(np.square(dataX.values - dataX.values.mean()))


# In[87]:


sumOfSqureOfY = np.sum(np.square(dataY.values - dataY.values.mean()))


# In[88]:


sqrtOfSSXY = np.sqrt( sumOfSqureOfX * sumOfSqureOfY )


# In[89]:


sqrtOfSSXY


# In[90]:


check_r = sumOfMultiplyOfMeanXY / sqrtOfSSXY


# In[91]:


check_r


# # $$y = a + bx$$
# 
# 
# ## $$b = r * (sdY/sdX)$$
# 
# ## $$a = meanY - b * meanX$$
# 

# In[92]:


b = r * (np.std(dataY.values) / dataX.values.std())


# In[93]:


b


# In[94]:


np.std(dataY.values)


# In[95]:


dataY.values.std()


# In[96]:


a = dataY.values.mean() - ( b * dataX.values.mean())


# In[97]:


a


# In[98]:


def Yprediction(x):
    return (a + (b * x))


# In[99]:


# Calculating Predicted Y
predictedY = dataX.apply(Yprediction)


# print(predictedY.head())?
# print(dataY.head())


# In[ ]:





# In[ ]:





# In[100]:


r == check_r


# In[101]:


print(type(dataY), type(predictedY))


# In[102]:


# >>> d = {'col1': [1, 2], 'col2': [3, 4]}
# >>> df = pd.DataFrame(data=d)
# >>> df

testdf = dataY
testdf['G2_Predicted'] = predictedY


# In[103]:



testdf.head()


# In[ ]:





# In[ ]:





# In[104]:


# type(testdf['G2_Predicted'].values)


# In[105]:


# SSR = np.sum( testdf['G2_Predicted'].values, testdf['G2'].values.mean() )
# testdf['G2'].values.mean()
# testdf['G2_Predicted'].values
# np.sum( np.square( testdf['G2_Predicted'].values, testdf['G2'].values.mean() ) )

# np.sum(np.square(testdf['G2_Predicted'].values - testdf['G2'].values.mean()))


# In[106]:


SSR = np.sum(np.square( np.subtract(testdf['G2_Predicted'].values , testdf['G2'].values.mean()) ))


# In[107]:


SSE = np.sum(np.square( np.subtract(testdf['G2'].values , testdf['G2_Predicted'].values.mean()) ))


# In[108]:


CoeffOfDetermination = (SSR / SSE)


# In[109]:


CoeffOfDetermination


# In[110]:


np.sqrt(CoeffOfDetermination)


# In[111]:


r


# In[ ]:





# In[ ]:





# In[ ]:





# In[112]:


plt.scatter(df.G1, df.G2)
plt.plot(df.G1, testdf[['G2_Predicted']])
plt.show()


# In[113]:


from sklearn import linear_model


reg = linear_model.LinearRegression()


# In[114]:


reg.fit(df[['G1']], df.G2)


# In[115]:


y_cap = reg.predict(df[['G1']])
reg.coef_


# In[116]:


# %matplotlib inline

plt.scatter(df.G1,df.G2,color='red', marker='o')
plt.plot(df.G1,y_cap)
plt.show()


# In[ ]:





# # Mean absolute error(MAE)
# 
# $$MAE = \frac{1}{n} \sum_{}\left\lvert{y- \hat{y}}\right\rvert$$
# 
# # Mean Squared Error
# 
# $$MSE = \frac{1}{n} \sum_{}{(y- \hat{y})^2}$$
# 
# # Root Mean Squared Error
# 
# $$RMSE = \sqrt{ \frac{1}{n} \sum_{}{(y- \hat{y})^2}}$$
# 

# In[117]:


# testdf.G2 - testdf.G2_Predicted


# In[118]:


MAE = ( (1/len(testdf.G2)) * np.sum(abs(np.subtract(testdf.G2 , testdf.G2_Predicted))) )


# In[119]:


MAE


# In[120]:


YY = testdf.G2
YY.head()
YYpredict = testdf.G2_Predicted
YYpredict.head()
# ( (1 / len(YY)) * ( np.sum(np.abs(np.subtract(YY,YYp)))) )
MSE = ( (1 / len(YY)) * ( np.sum(np.square(np.subtract(YY,YYpredict)))) )


# In[121]:


MSE


# In[122]:


RMSE = np.sqrt(MSE)


# In[123]:


RMSE


# In[124]:


from sklearn.metrics import mean_squared_error

from math import sqrt

rmse = sqrt(mean_squared_error(YY, YYpredict))

print(rmse)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




