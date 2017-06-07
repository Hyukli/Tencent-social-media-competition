#-*- coding:utf-8 -*-
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier

user=pd.read_csv('database/user.csv')
print user[:50]
print user['age'].value_counts()
#30万没有（留下了，分段）
print user['gender'].value_counts()
#28万没有（留下来，预测）
print user['education'].value_counts()
#70万没有（二分有无）
print user['marriageStatus'].value_counts()
#113万没有（二分有无）
print user['haveBaby'].value_counts()
#227万没有（二分有无）
print user['hometown'].value_counts()
#96万没有（二分有无）
print user['residence'].value_counts()
#23万没有（留下来，预测）