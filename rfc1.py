#-*- coding:utf-8 -*-
import pandas as pd
train=pd.read_csv('database/train.csv')
test=pd.read_csv('database/test.csv')

#print train.info()
#print test.info()
selected_features=['clickTime','creativeID','userID','positionID','connectionType','telecomsOperator']
X_train=train[selected_features]
X_test=test[selected_features]
print X_train.info()
print X_test.info()
y_train=train['label']
from sklearn.feature_extraction import DictVectorizer
dict_vec=DictVectorizer(sparse=False)
print '开始训练特征向量'
X_train=dict_vec.fit_transform(X_train.to_dict(orient='record'))
print '完成训练特征向量'

X_test=dict_vec.transform(X_test.to_dict(orient='record'))
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()

print '开始训练分类器'
rfc.fit(X_train,y_train)
print '完成训练特征向量'

print '开始测试'
rfc_y_predict=rfc.predict(X_test)
print '完成测试'

print '最后赋值开始'
rfc_submission=pd.DataFrame({'instanceID':test['instanceID'],'prob':rfc_y_predict})
#rfc_submission=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':rfc_y_predict})


rfc_submission.to_csv('database/rfc_submission.csv',index=False)
print '最后赋值结束'
