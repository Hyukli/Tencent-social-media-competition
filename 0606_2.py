#-*- coding:utf-8 -*-
import pandas as pd
import sklearn.preprocessing as preprocessing
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import numpy as np
print '开始读取文件'
user=pd.read_csv('database/user.csv')
train=pd.read_csv('database/train.csv')
y_train=train['label']
test=pd.read_csv('database/test.csv')
print '结束读取文件'

#把user表其中的0均改为空
user['age'].replace(0,np.nan,inplace=True)

##########预测年龄函数##########
def set_missing_ages(df):

    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['age','education', 'marriageStatus', 'haveBaby','hometown']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.age.notnull()].as_matrix()
    unknown_age = age_df[age_df.age.isnull()].as_matrix()

    # y即目标年龄,其中第一个：代表所有行，0代表所有的第一列
    y = known_age[:, 0]

    # X即特征属性值，其中第一个：代表所有行，1：代表除了第一列之外的所有列
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])

    # 用得到的预测结果填补原缺失数据
    df.loc[ (df.age.isnull()), 'age' ] = predictedAges

    return df, rfr
##########二分属性函数、共4种##########
def set_erfen_type(df):
    df.loc[(df.education.notnull()), 'education'] = "Yes"
    df.loc[(df.education.isnull()), 'education'] = "No"
    df.loc[(df.marriageStatus.notnull()), 'marriageStatus'] = "Yes"
    df.loc[(df.marriageStatus.isnull()), 'marriageStatus'] = "No"
    df.loc[(df.haveBaby.notnull()), 'haveBaby'] = "Yes"
    df.loc[(df.haveBaby.isnull()), 'haveBaby'] = "No"
    df.loc[(df.hometown.notnull()), 'hometown'] = "Yes"
    df.loc[(df.hometown.isnull()), 'hometown'] = "No"
    return df

user, rfr = set_missing_ages(user)
user['gender'].replace(0,np.nan,inplace=True)
user['education'].replace(0,np.nan,inplace=True)
user['marriageStatus'].replace(0,np.nan,inplace=True)
user['haveBaby'].replace(0,np.nan,inplace=True)
user['hometown'].replace(0,np.nan,inplace=True)
user['residence'].replace(0,np.nan,inplace=True)
user = set_erfen_type(user)

#########平铺除了age的其他6属性######
dummies_gender = pd.get_dummies(user['gender'], prefix= 'gender')
dummies_education = pd.get_dummies(user['education'], prefix= 'education')
dummies_marriageStatus = pd.get_dummies(user['marriageStatus'], prefix= 'marriageStatus')
dummies_haveBaby = pd.get_dummies(user['haveBaby'], prefix= 'haveBaby')
dummies_hometown = pd.get_dummies(user['hometown'], prefix= 'hometown')
dummies_residence = pd.get_dummies(user['residence'], prefix= 'residence')

user = pd.concat([user, dummies_gender, dummies_education, dummies_marriageStatus, dummies_haveBaby,dummies_hometown,dummies_residence], axis=1)
user.drop(['gender', 'education', 'marriageStatus', 'haveBaby', 'hometown', 'residence'], axis=1, inplace=True)

######把age范围变得正常
scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(user['age'])
user['age_scaled'] = scaler.fit_transform(user['age'], age_scale_param)

#分别链接test/train和user两个表
user_train=pd.merge(train,user,how='left')
user_test=pd.merge(test,user,how='left')

#挑选合适的特征,这里挑选了所有train/user表的特征
#selected_feas=['clickTime','userID','creativeID','positionID','connectionType','telecomsOperator','age_.*','gender_.*','education_.*','marriageStatus_.*']
#user_train=user_train[selected_feas]
user_train=user_train.filter(regex='clickTime|userID|creativeID|positionID|connectionType|telecomsOperator|age_.*|gender_.*|education_.*|marriageStatus_.*')
#user_test=user_test[selected_feas]
user_test=user_test.filter(regex='clickTime|userID|creativeID|positionID|connectionType|telecomsOperator|age_.*|gender_.*|education_.*|marriageStatus_.*')
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

rfc_submission.to_csv('database/0606_user_sep_submission.csv',index=False)
print '最后赋值结束'
