import numpy as np
import pandas
import xgboost as xgb
import matplotlib.cm as cm
from sklearn.ensemble import RandomForestRegressor
from itertools import *
import lightgbm as lgb
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from sklearn.metrics import r2_score
from tpot import TPOTRegressor
from sklearn.manifold import TSNE
import binascii

###
##  Harish 
## one of Kaggle competetion 
###
category_data=['X0','X1','X2','X3','X4','X5','X6','X8']

def xrange(x):
    return iter(range(x))

def tpotBestModelSearch():
    tpot = TPOTRegressor(generations=5,population_size=20,verbosity=2)
    tpot.fit(x_train,y_train)
    print(tpot.score(x_validation,y_validation))

def dataEncoder(df_train,df_test):
    for c in df_train.columns:
        if df_train[c].dtype == 'object':
            lbl= LabelEncoder()
            lbl.fit(list(df_train[c].values) + list(df_test[c].values))
            df_train[c]= lbl.transform(list(df_train[c].values))
            df_test[c] = lbl.transform(list(df_test[c].values))
    return df_train,df_test


def pre_processing_data(df_train,df_test) :
    cols = df_train.select_dtypes([np.number]).columns
    std = df_train[cols].std()
    cols_to_drop = std[std<0.02].index
    df_train = df_train.drop(cols_to_drop, axis=1)
    df_test = df_test.drop(cols_to_drop,  axis=1) 
    return df_train,df_test

def randomForestModel(x_train,y_train,x_test):
    rfModel = RandomForestRegressor()
    rfModel.fit(x_train,y_train)
    importances = rfModel.feature_importances_
    feature_labels=x_train.columns
    indices = np.argsort(importances)[::-1]
    order_features = []
    order_importances=[]
    for f in range(x_train.shape[1]):
        order_features.append(feature_labels[f])
        order_importances.append(importances[indices[f]])
    return rfModel,x_test

def xgboostFitModel(x_train,y_train,x_test):
    y_mean=np.mean(y_train)
    xgb_params = {
        'eta' : 0.05,
        'max_depth': 2,
        'subsample' : 0.65,
        'objective': 'reg:linear',
        'eval_metric': 'mae',
        'base_score':y_mean,
        'silent':1
    }
    dtrain = xgb.DMatrix(x_train,y_train)
    cv_result = xgb.cv(xgb_params,dtrain,nfold=5,
                        num_boost_round=200,
                        early_stopping_rounds=5,
                        verbose_eval=10,
                        show_stdv=False)
    num_boost_rounds=len(cv_result)
    model=xgb.train(dict(xgb_params,silent=1),dtrain,num_boost_round=num_boost_rounds)
    dtest = xgb.DMatrix(x_test)
    return model,dtest  



def completeModelPrediction(df_train,df_test):
    
    xtrain,xtest = dataEncoder(df_train,df_test)
    x_train = xtrain.drop(['ID','y'],axis =1)
    y_train = train_df['y'].values.astype(np.float32)
    
    x_test  = xtest.drop(['ID'],axis=1)
    
    tpot = TPOTRegressor(generations=5,population_size=5,verbosity=2)
    tpot.fit(x_train,y_train)
    pred = tpot.predict(x_test)
    
    #model,dtest = xgboostFitModel(x_train,y_train,x_test)
    #pred = model.predict(dtest)
    newdf = formOutput(pred,xtest,0)
    return newdf

def category_data_prediction_rf(df_train,df_test):
    cols_to_get_train = ['X0','X1','X2','X3','X4','X5','X6','X8','ID','y']
    cols_to_get_test = ['X0','X1','X2','X3','X4','X5','X6','X8','ID']

    df_train = df_train[cols_to_get_train]
    df_test = df_test[cols_to_get_test]

    xtrain,xtest = dataEncoder(df_train,df_test)
    y_train = xtrain['y'].values.astype(np.float32)
    x_train = xtrain.drop(['ID','y'],axis=1)
    x_test  = xtest.drop(['ID'],axis=1)

    #model,dtest = randomForestModel(x_train,y_train,x_test)
    #pred = model.predict(dtest)
    tpot = TPOTRegressor(generations=5,population_size=5,verbosity=2)
    tpot.fit(x_train,y_train)
    pred = tpot.predict(x_test)
    

    newdf = formOutput(pred,xtest,0)
    return newdf

def groupedX0_prediction(df_train,df_test):
    #X0 grp
    X0_1 = ['bc','az']
    X0_2 = ['ac','am','l','b','aq','u','ad','e','al','s','n','y', 't',  'ai',  'k' , 'f',  'z',  'o', 'ba', 'm','q']
    X0_3 = ['d', 'ay', 'h', 'aj' ,'v','ao','aw','ae', 'ag', 'bb', 'an', 'p', 'av']
    X0_4 = ['c','ax','x','j','w','i','ak','g','at','ab','af','r', 'as',  'a',  'ap' , 'au']
    
    X0_N = ['ae', 'ag', 'bb', 'an', 'p', 'av']

    X0_grp = np.array([X0_1,X0_2,X0_3,X0_4])
    output_df = pandas.DataFrame({})
    i=0
    for grp in X0_grp:
        i=i+1
        xtrain = df_train.loc[df_train['X0'].isin(grp)]
        xtest = df_test.loc[df_test['X0'].isin(grp)]
        xtrain = xtrain.drop(['X0'],axis =1)
        xtest =  xtest.drop(['X0'],axis=1)
        xtrain,xtest = dataEncoder(xtrain,xtest)

        y_train = xtrain['y'].values.astype(np.float32)
        x_train = xtrain.drop(['ID','y'],axis =1)
        x_test  = xtest.drop(['ID'],axis=1)
        #model,dtest = xgboostFitModel(x_train,y_train,x_test)
        #pred = model.predict(dtest)
        tpot = TPOTRegressor(generations=5,population_size=5,verbosity=2)
        tpot.fit(x_train,y_train)
        pred = tpot.predict(x_test)
        
        newdf = formOutput(pred,xtest,i)
        output_df = output_df.append(newdf)
    return output_df

def binaryDataprediction(df_train,df_test):
    df_train = df_train.drop(category_data,axis=1)       
    df_test  = df_test.drop(category_data,axis=1)
    
    x_train = df_train.drop(['ID','y'],axis=1)
    y_train = df_train['y'].values.astype(np.float32)
    x_test  = df_test.drop(['ID'],axis=1)
    #model,dtest = xgboostFitModel(x_train,y_train,x_test)
    #pred = model.predict(dtest)
    tpot = TPOTRegressor(generations=5,population_size=5,verbosity=2)
    tpot.fit(x_train,y_train)
    pred = tpot.predict(x_test)
    
    newdf = formOutput(pred,df_test,0)
    return newdf


def formOutput(pred,df_test,index):
    y_pred=[]
    for i,predict in enumerate(pred):
        y_pred.append(str(round(predict,12)))
    y_pred=np.array(y_pred)
    output = pandas.DataFrame({'ID':df_test['ID'].astype(np.int32),'y':y_pred})
    return output

def finalPred(pred_complete,pred_category, pred_X0_grp,pred_binary,df_train,df_test):
    pred_binary['y_comp'] = pred_complete['y']
    pred_binary['y_cat'] = pred_category['y']
    pred_binary['y_x0'] = pred_X0_grp['y']
    return pred_binary


def getData():
    train_url="train.csv"
    test_url="test.csv"
    train_df = pandas.read_csv(train_url)
    test_df=pandas.read_csv(test_url)
    train_df,test_df = pre_processing_data(train_df,test_df)
    return train_df,test_df



#####################
#####     MAIN ######
#####################


#Get full trained model on xgbboost
print ("***********************")
print ("completeModelPrediction")

train_df,test_df = getData()
pred_complete = completeModelPrediction(train_df,test_df)

print ("***********************")
print ("category_data_prediction_rf")
train_df,test_df = getData()
pred_category = category_data_prediction_rf(train_df,test_df)

print ("***********************")
print ("groupedX0_prediction")
train_df,test_df = getData()
pred_X0_grp = groupedX0_prediction(train_df,test_df)

print ("***********************")
print ("binaryDataprediction")
train_df,test_df = getData()
pred_binary = binaryDataprediction(train_df,test_df)

print ("***********************")
print ("finalPred")
train_df,test_df = getData()
predFinal = finalPred(pred_complete,pred_category,pred_X0_grp,pred_binary,train_df,test_df)

#put to submission file
predFinal.to_csv(path_or_buf="submission_out.csv",index=False)
