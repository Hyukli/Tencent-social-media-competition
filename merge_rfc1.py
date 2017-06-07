#-*- coding:utf-8 -*-
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
print '开始读取文件'
train=pd.read_csv('database/train.csv')
y_train=train['label']
test=pd.read_csv('database/test.csv')
user=pd.read_csv('database/user.csv')
print '结束读取文件'
#ad=pd.read_csv('database/ad.csv')
#position=pd.read_csv('database/position.csv')
#user_installedapps=pd.read_csv('database/user_installedapps.csv')
#user_app_actions=pd.read_csv('database/user_app_actions.csv')
#app_categories=pd.read_csv('database/app_categories.csv')


####现在我在填充user表####
print '开始填充user表'
#平均值填充用户年龄
#print user['age'].mean()
user['age'].replace(0,22,inplace=True)
#众数填充性别
#print user['gender'].value_counts()
user['gender'].replace(0,1,inplace=True)
#众数填充教育程度
#print user['education'].value_counts()
user['education'].replace(0,1,inplace=True)
#众数填充婚姻状况
#print user['marriageStatus'].value_counts()
user['marriageStatus'].replace(0,1,inplace=True)
#众数填充子女状况
#print user['haveBaby'].value_counts()
user['haveBaby'].replace(0,1,inplace=True)
#众数填充籍贯状况
#print user['hometown'].value_counts()
user['hometown'].replace(0,1901,inplace=True)
#众数填充住址状况
#print user['residence'].value_counts()
user['residence'].replace(0,1901,inplace=True)

print '结束填充user表'

#分别链接test/train和user两个表
user_train=pd.merge(train,user,how='left')
user_test=pd.merge(test,user,how='left')
#挑选合适的特征
selected_feas=['clickTime','creativeID','positionID','connectionType','telecomsOperator','age','gender','education','marriageStatus']
user_train=user_train[selected_feas]
user_test=user_test[selected_feas]
dict_vec=DictVectorizer(sparse=False)
user_train=dict_vec.fit_transform(user_train.to_dict(orient='record'))
user_test=dict_vec.transform(user_test.to_dict(orient='record'))
#使用随机森林训练
print '开始训练分类器'
rfc=RandomForestClassifier()
rfc.fit(user_train,y_train)
print '开始训练结果集合'
rfc_y_predict=rfc.predict(user_test)
print '最后赋值开始'
rfc_submission=pd.DataFrame({'instanceID':test['instanceID'],'prob':rfc_y_predict})
#rfc_submission=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':rfc_y_predict})
rfc_submission.to_csv('database/join_rfc_submission1.csv',index=False)
print '最后赋值结束'
