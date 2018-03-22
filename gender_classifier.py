# -*- coding: utf-8 -*-
__author__ = 'shark'
__date__ = '2018/3/22, 下午3:19'
__filename__ = 'gender'



X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']



from  sklearn.metrics import  accuracy_score
from  sklearn.neighbors import  KNeighborsClassifier
from  sklearn.svm import  SVC
from  sklearn.naive_bayes import  GaussianNB
from  sklearn.ensemble import  RandomForestClassifier, AdaBoostClassifier
import  numpy as np



kclf = KNeighborsClassifier()
kclf.fit(X,Y)

sclf = SVC()
sclf.fit(X,Y)

nbclf = GaussianNB()
nbclf.fit(X,Y)

rclf = RandomForestClassifier()
rclf.fit(X,Y)


# test
pred_k = kclf.predict(X)
acc_k = accuracy_score(Y,pred_k)

pred_s = sclf.predict(X)
acc_s = accuracy_score(Y,pred_s)

pred_b = nbclf.predict(X)
acc_b = accuracy_score(Y,pred_b)

pred_r = rclf.predict(X)
acc_r = accuracy_score(Y,pred_r)


index = np.argmax([acc_k,acc_s,acc_b,acc_r])
print([acc_k,acc_s,acc_b,acc_r])
df = {0: 'KNeighborsClassifier', 1:'SVC',2:'GaussianNB',3:'RandomForestClassifier'}

print(df[index])




