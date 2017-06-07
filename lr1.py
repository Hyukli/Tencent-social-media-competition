#-*- coding:utf-8 -*-
import pandas as pd
train=pd.read_csv('database/train.csv')
test=pd.read_csv('database/test.csv')

#print train.info()
#print test.info()
selected_features=['clickTime','creativeID','userID','positionID','connectionType','telecomsOperator']
X_train=train[selected_features]
X_test=test[selected_features]
#print X_train.info()
#print X_test.info()
y_train=train['label']
from sklearn.feature_extraction import DictVectorizer
dict_vec=DictVectorizer(sparse=False)
print '开始训练特征向量'
X_train=dict_vec.fit_transform(X_train.to_dict(orient='record'))
print '完成训练特征向量'

X_test=dict_vec.transform(X_test.to_dict(orient='record'))
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()

print '开始训练分类器'
lr.fit(X_train,y_train)
print '完成训练特征向量'

print '开始测试'
rfc_y_predict=lr.predict_proba(X_test)
print test.info()
print '完成测试'

print rfc_y_predict[:10]

print '最后赋值开始'
rfc_submission=pd.DataFrame({'instanceID':test['instanceID'],'prob':rfc_y_predict[1]})
#rfc_submission=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':rfc_y_predict})

print rfc_submission[:10]

rfc_submission.to_csv('database/lr1_submission.csv',index=False)
print '最后赋值结束'
