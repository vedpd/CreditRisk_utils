# Building base data for modelling
import sklearn
import pandas as pd
import numpy as np
import gc
from xgboost.sklearn import XGBClassifier

from sklearn.metrics import roc_auc_score

#reading dataset
train_test_val_all_data= "s3://lending-data-science/ved_prakash_dwivedi/PP_NS_Final_Model/Version2_location_dropped/Model_data/train_test_val/all_data_with_train_test_val/"

dataset = pd.read_parquet(train_test_val_all_data)

#reading train and test
train_df = dataset[dataset['Sample_group']=="train"]
test_df = dataset[dataset['Sample_group']=="test"]
# val_df = dataset[dataset['Sample_group']=="val"]

#memory management
import gc
del dataset
gc.collect()

#getting list of columns for modelling
all_cols_paytm = pd.DataFrame(data=train_df.columns, columns =['column_names'])

#validate the distribution
train_df.target.value_counts(), test_df.target.value_counts()

#selecting only 0 and 1 as target and dropping 2 as indeterminate
train_set=train_df[train_df['target']!=2]
test_set=test_df[test_df['target']!=2]

#removing column cols that are not to be used in modelling
common_cols= ['lead_id', 'customer_id', 'lead_disbursed_month',
              'cum_max_dpd_3mob', 'applied_date', 'asof', 'min_txn_month',
              'transaction_amount', 'IsAppDisbTxnSameMonth', 'IsDisbTxnSameMonth','Sample','Sample_group', 
              'target']

feat_selected_modelling = [item for item in train_set.columns if item not in common_cols]

len(feat_selected_modelling)

#convert all non int cols to int cols
# cols_obj= ['MYC_credit_card_payment_gmv_1m',
# 'MYC_credit_card_payment_gmv_12m', 'MYC_credit_card_payment_gmv_2m', 
# 'MYC_credit_card_payment_gmv_6m', 'MYC_health_gmv_12m', 'MYC_health_gmv_6m',
# 'MYC_health_gmv_1m', 'MYC_health_gmv_3m', 'MYC_health_gmv_2m', 'MYC_credit_card_payment_gmv_3m', 
# 'MYC_credit_card_payment_gmv_9m', 'MYC_health_gmv_9m']
# train_set[cols_obj]=train_set[cols_obj].astype(int)

# train_set[cols_obj].dtypes


#Creating X_dev, Y_dev, X_test, Y_test as separate entities
X_dev = train_set.drop(common_cols,axis=1)[feat_selected_modelling]
Y_dev = train_set[[common_cols[-1]]]

X_test = test_set.drop(common_cols,axis=1)[feat_selected_modelling]
Y_test = test_set[[common_cols[-1]]]


#Experiment 1: 30 iter : hyperopt
from hyperopt import tpe, hp, fmin, STATUS_OK,Trials
from hyperopt.pyll.stochastic import sample
from hyperopt.pyll.base import scope


from hyperopt import STATUS_OK
from timeit import default_timer as timer
from xgboost.sklearn import XGBClassifier
import xgboost as xgb

from hyperopt import tpe, hp, fmin, STATUS_OK,Trials
from hyperopt.pyll.stochastic import sample
from hyperopt.pyll.base import scope


from hyperopt import STATUS_OK
from timeit import default_timer as timer
from xgboost.sklearn import XGBClassifier
import xgboost as xgb

def objective(space):
    """Objective function for Gradient Boosting Machine Hyperparameter Optimization"""
    global  ITERATION
    ITERATION += 1
    
    start = timer()
    xgb_params = space
       
    print('Params selected: ',xgb_params)
    
    
    model = XGBClassifier(**xgb_params)#, monotone_constraints = final_feature_constrain_dic)
    
    model.fit(X_dev, Y_dev,  
              eval_set=[(X_dev,Y_dev), (X_test, Y_test)],
              verbose=True)
    
    run_time = timer() - start
    
    auc_test = max(model.evals_result()['validation_1']['auc'])
    index = model.evals_result()['validation_1']['auc'].index(max(model.evals_result()['validation_1']['auc']))
    auc_train = model.evals_result()['validation_0']['auc'][index]
    
    
    n_estimators = model.best_iteration + 1 #same as index+1
    
     # Creating feature importance dataframe
    feat_df = pd.DataFrame({'Features': X_dev.columns, 'Importance': model.feature_importances_})
    feat_df.sort_values('Importance', ascending=False, inplace=True)
    feat_df['cumul_fea_imp'] = feat_df['Importance'].cumsum()
    final_features_temp = feat_df[feat_df['cumul_fea_imp']<=0.90]['Features'].tolist()
    fea_90_cnt = len(final_features_temp)
    print("fea_90_cnt:", fea_90_cnt)
    
    
    # Extract the best score
    best_score = auc_test
    
    # Loss must be minimized
    loss = 1 - best_score

    # Write to the csv file ('a' means append)
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([auc_train,auc_test,loss, xgb_params, ITERATION,n_estimators,run_time, fea_90_cnt])
    
    # Dictionary with information for evaluation
    return {'loss': loss,
            'params': xgb_params, 
            'iteration': ITERATION,
            'estimators': n_estimators, 
            'train_time': run_time,
            'status': STATUS_OK}

#define search space
space = {
   'n_estimators': hp.choice('n_estimators', [200]),
    'early_stopping_rounds': 10,
    'learning_rate': hp.quniform('learning_rate', 0.05,0.5,0.05),
    'max_depth': hp.choice('max_depth', [2,3,4,5,6]),
    'subsample': hp.choice('sub_sample', [0.7,0.8,0.9]),
    'min_child_weight': hp.quniform('min_child_weight', 50,1000,10),
    'gamma': hp.quniform('gamma',5,100,1),
    'reg_alpha': hp.quniform('reg_alpha', 0.01,10, 0.05),
    'eval_metric': 'auc',
    'objective': 'binary:logistic',
    'nthread': 100,
    'booster': 'gbtree',
    'importance_type':'gain',
    'random_state': 101,
    'missing': -999999.0    
    }

#initiate trial count
trail_count=0
print(trail_count)

# track file for model runs
from time import time 
trials = Trials()
import csv
trail_count+=1
print(trail_count)

# File to save first results
out_file = f'./Hyperopt_results/xgb_trials_30iter_stage1_NS_{str(trail_count)}.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(['train_auc','test_auc','loss', 'xgb_params', 'iteration', 'estimators','train_time', 'fea_90_cnt'])
of_connection.close()





# initiate the bayes run
bayes_trials = Trials()

# Global variable
global  ITERATION

ITERATION = 0

# Run optimization
best = fmin(fn = objective, 
            space = space, 
            algo = tpe.suggest, 
            max_evals = 30, 
            trials = bayes_trials, 
            verbose=True,
            show_progressbar=True)

