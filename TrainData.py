
class Train_Data():
    
    def __init__(self):
        # ----Other imports ----------------
        from sklearn.model_selection import KFold,StratifiedKFold
        from sklearn.calibration import CalibratedClassifierCV
        from imblearn.under_sampling import RandomUnderSampler
        from sklearn import model_selection, preprocessing
        from sklearn.metrics import roc_curve, auc
        import matplotlib.pyplot as plt   
        from scipy import interp
        import seaborn as sn
        import pandas as pd
        import numpy as np
        import warnings
        import sys
        import warnings
        warnings.simplefilter('ignore',DeprecationWarning)
        pd.options.mode.chained_assignment=None
        ros=RandomUnderSampler()

    def train_model(self,x_data=[],y_data=[],con_cols=[],cat_cols=[],model=[],imputer=[],cvsplit=4,rstate=101,misper=[]):
        import warnings
        warnings.simplefilter('ignore',DeprecationWarning)
        import numpy as np
        import xgboost as xgb
        from sklearn.model_selection import GridSearchCV as GSCV
        from sklearn.model_selection import KFold,StratifiedKFold
                # ----model imports ----------------
        from sklearn.ensemble import AdaBoostClassifier as ABC
        from sklearn.linear_model import LogisticRegression as LR
        from sklearn.ensemble import RandomForestClassifier as RFC
        import xgboost as xgb


        #-----------------Selecting the imputer------------------------------------------
        Imputed_Data=self.Impute_the_data(imputer=imputer,x_data=x_data,y_data=y_data,con_cols=con_cols,cat_cols=cat_cols,misper=misper)

        train_y=Imputed_Data['train_y']
        train_x=Imputed_Data['train_x']
        X_resampled=Imputed_Data['X_resampled']
        y_resampled=Imputed_Data['y_resampled']

        #-----------------Selecting the model to run-------------------------------------
        
        if model == "xgboost":
            
            paramGrid={
                        'max_depth':[5,10],
                        'min_child_weight':np.arange(1,9,1),
                        'gamma':np.arange(0,1,0.001),
                        'subsample':np.arange(0.1,0.9,0.05),
                        'colsample_bytree':np.arange(0.1,0.9,0.05),
                        'n_estimator':[50,100,200],
                       'objective':['binary:logistic','binary:logitraw'],
                       'learning_rate' : [0.001,0.01,0.1]}
    
            xgb_params={'eval_metric':'auc'}
            model_run=xgb.XGBClassifier()
            gridsearch=GSCV(model_run,paramGrid,verbose=1,fit_params=xgb_params,cv=KFold(n_splits=cvsplit,random_state=rstate).get_n_splits([train_x,train_y]))
            gridsearch.fit(train_x,train_y)
            xgb_params=dict(gridsearch.best_params_)
            params=xgb_params

        elif model == "adaboost":
            model_run=ABC()
            paramGrid={'learning_rate' : [0.001,0.01,0.1],'n_estimators':[50,100,200]}
            gridsearch=GSCV(model_run,paramGrid,verbose=1,cv=KFold(n_splits=cvsplit,random_state=rstate).get_n_splits([train_x,train_y]))
            gridsearch.fit(train_x,train_y)
            ada_params=dict(gridsearch.best_params_)
            params=ada_params

            
        elif model == "logreg":
            model_run=LR()
            params=[]
            
        elif model == "randomforest":
            model_run=RFC()                
            params=[]
            
        elif model == "lightgbm":
            print ('lightgbm still not configured\n')
            sys.exit()

        Output=self.run_the_model(model_run,model,X_resampled,y_resampled,train_x,params,rstate,cvsplit)
      

        return {'model':Output['model'],'Acc_vals': Output['Acc_vals'],'Mean_vals':Output['Mean_vals'],'dataset' : Output['dataset'],'modeltype':Output['modeltype']}
    
    def run_the_model(self,model,modeltype,X_resampled,y_resampled,x_filtered,params,rstate,cvsplit):
        from sklearn.model_selection import train_test_split as tts
        from sklearn.model_selection import KFold,StratifiedKFold
        from sklearn.calibration import CalibratedClassifierCV
        from imblearn.under_sampling import RandomUnderSampler
        from sklearn import model_selection, preprocessing
        from sklearn.metrics import roc_curve, auc
        import matplotlib.pyplot as plt   
        from scipy import interp
        import seaborn as sn
        import pandas as pd
        import numpy as np
        import warnings
        import sys
        import xgboost as xgb
        warnings.simplefilter('ignore',DeprecationWarning)
        
        X,X_val,y,y_val=tts(X_resampled,y_resampled,test_size=0.25,random_state=rstate)
     
        Test_acc_vals=[]
        Val_acc_vals=[]
        models=[]
        
        kf=StratifiedKFold(n_splits=cvsplit,shuffle=True,random_state=rstate)
        
        train_x=[]
        train_y=[]
        test_x=[]
        test_y=[]
        val_x=[]
        val_y=[]
        
        test_fpr=[]
        test_tpr=[]
        
        val_fpr=[]
        val_tpr=[]
        
        test_mean_fpr=np.linspace(0,1,100)
        val_mean_fpr=np.linspace(0,1,100)
        
        test_tprs=[]
        val_tprs=[]
        
        
        for train_index,test_index in kf.split(X,y):
            X_train,X_test=X[train_index],X[test_index]
            y_train,y_test=y[train_index],y[test_index]

            
            if modeltype=='xgboost':
                
                dtrain=xgb.DMatrix(X_train,y_train,feature_names=x_filtered.columns.values)
            
                model=xgb.XGBClassifier()
                model=xgb.train(dict(params,silent=0),dtrain,num_boost_round=100)
            
                dtest=xgb.DMatrix(X_test,y_test,feature_names=x_filtered.columns.values)
            
                y_pred=model.predict(dtest)
                fpr, tpr, thresholds = roc_curve(y_test, y_pred)
                roc_auc = auc(fpr, tpr)
                Test_acc_vals.append(roc_auc)
            
                test_fpr.append(fpr)
                test_tpr.append(tpr)
            
                test_tprs.append(interp(test_mean_fpr,fpr,tpr))
                test_tprs[-1][0]=0.0
            
                dval=xgb.DMatrix(X_val,y_val,feature_names=x_filtered.columns.values)
            
                y_pred=model.predict(dval)
                fpr, tpr, thresholds = roc_curve(y_val, y_pred)
                roc_auc = auc(fpr, tpr)
                Val_acc_vals.append(roc_auc)
                val_fpr.append(fpr)
                val_tpr.append(tpr)
                
                val_tprs.append(interp(val_mean_fpr,fpr,tpr))
                val_tprs[-1][0]=0.0
                
                models.append(model)
                train_x.append(X_train)
                test_x.append(X_test)
                
                
                train_y.append(y_train)
                test_y.append(y_test)

            else :
                if modeltype=='adaboost':
                    from sklearn.ensemble import AdaBoostClassifier as ABC
                    model=ABC(learning_rate=params['learning_rate'],n_estimators=params['n_estimators'])
                elif modeltype=='logreg':
                    from sklearn.linear_model import LogisticRegression as LR
                    model=LR()
                elif modeltype=='randomforest':
                    from sklearn.ensemble import RandomForestClassifier as RFC
                    model=RFC()

                model.fit(X_train,y_train)
                y_pred=model.predict_proba(X_test)
                
                fpr, tpr, thresholds = roc_curve(y_test, y_pred[:,1])
                roc_auc = auc(fpr, tpr)
                Test_acc_vals.append(roc_auc)
                
                test_fpr.append(fpr)
                test_tpr.append(tpr)
                
                test_tprs.append(interp(test_mean_fpr,fpr,tpr))
                test_tprs[-1][0]=0.0
            
                
                y_pred=model.predict_proba(X_val)
                fpr, tpr, thresholds = roc_curve(y_val, y_pred[:,1])
                roc_auc = auc(fpr, tpr)
                Val_acc_vals.append(roc_auc)
                val_fpr.append(fpr)
                val_tpr.append(tpr)
                
                val_tprs.append(interp(val_mean_fpr,fpr,tpr))
                val_tprs[-1][0]=0.0
                
                models.append(model)
                train_x.append(X_train)
                test_x.append(X_test)
                
                
                train_y.append(y_train)
                test_y.append(y_test)

                
            val_x.append(X_val)
            val_y.append(y_val)
            
            try:
                test_mean_tprs=np.mean(np.float64(test_tprs),axis=0)
                test_mean_tprs[-1]=np.float64(1)
                val_mean_tprs=np.mean(np.float64(val_tprs),axis=0)
                val_mean_tprs[-1]=np.float64(1)
            except:
                val_mean_tprs=0
                test_mean_tprs=0         
            

        return {'model':models,
                'Acc_vals': {
                'Test': {'Acc':Test_acc_vals,'FPR':test_fpr,'TPR':test_tpr},
                'Val':{'Acc':Val_acc_vals,'FPR':val_fpr,'TPR':val_tpr}},
                'Mean_vals':{
                'Test':{'Acc':np.mean(Test_acc_vals),'FPR':test_mean_fpr,'TPR':test_mean_tprs},
                'Val':{'Acc':np.mean(Val_acc_vals),'FPR':val_mean_fpr,'TPR':val_mean_tprs}},
                'dataset' : {
                'x_train':train_x,'y_train': train_y,'x_test':test_x,'y_test':test_y,'x_val':val_x,'y_val':val_y},
                'modeltype':modeltype}
    
    def Impute_the_data(self,imputer="MEANMEDIAN",x_data=[],y_data=[],con_cols=[],cat_cols=[],misper=[]):

        import pandas as pd
        from imblearn.under_sampling import RandomUnderSampler
        import warnings
        warnings.simplefilter('ignore',DeprecationWarning)
               
        if imputer == "MICE":

            #savenane for the imputed file for reuse

            x_filtered_savename='./Data/x_filtered_'+imputer+'_'+str(misper)+'.csv'

            #try to see if the old file is saved and use it 

            try:
                x_filtered=pd.read_csv(x_filtered_savename)
                x_filtered=pd.DataFrame(x_filtered,columns=x_data.columns)
                print "Loaded from presaved file"

            #If the old file is non-existent, process the file and save new file

            except:
                print "Could not find the saved file. Generating new one with MICE imputation.\n"
                
                from fancyimpute import  MICE
                
                impute=MICE()
                x_filtered=x_data
                if int(misper)>0:
                    x_filtered=impute.complete(x_filtered)
                x_filtered=pd.DataFrame(x_filtered,columns=x_data.columns)
                x_filtered.to_csv(x_filtered_savename,index=False)

        ros=RandomUnderSampler()
    
        X_resampled,y_resampled =ros.fit_sample(x_filtered.values,y_data.values.ravel())
        
        train_x=pd.DataFrame(X_resampled,columns=x_data.columns)
        train_y=pd.DataFrame(y_resampled,columns=y_data.columns)

        return {'train_y':train_y,'train_x':train_x,'X_resampled':X_resampled,'y_resampled':y_resampled}
    
    def plot_FI_AUC(self,Output=[],modeltype=[]):
        
        model=Output['model']
        featureImportance=model.get_fscore()
        features=pd.DataFrame()
        features['features']=featureImportance.keys()
        features['importance']=featureImportance.values()
        features.sort_values(by=['importance'],ascending=False,inplace=True)
        fig,ax=plt.subplots()
        fig.set_size_inches(20,10)
        plt.xticks(rotation=90)
        sn.barplot(data=features.head(15),x="importance",y="features",ax=ax,orient="h",color="teal")
        savedir='./Figures/FI_'+modeltype+'_.png'
        ax=plt.gca()
        plt.savefig(savedir)
        plt.close('all')
        
        test_x=Output['dataset']['x_test']
        test_y=Output['dataset']['y_test']
        
        model=Output['calmodel']
                
        if modeltype == 'xgboost':
            import xgboost as xgb
            dtest=xgb.DMatrix(test_x,test_y,feature_names=test_x.columns.values)
        
        
        y_pred=model.predict_proba(test_x)
        lw=2
        fpr, tpr, thresholds = roc_curve(test_y, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.figure(1)
        plt.plot(fpr, tpr, color='gray', linestyle='-.',lw=lw,label='Test ROC (area = %0.4f)' %(roc_auc))
        plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='r',label='Luck')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        fig = plt.gcf()
        fig.set_size_inches(6,4)
        savedir='./Figures/AUC_'+modeltype+'_.png'
        ax=plt.gca()
        plt.savefig(savedir)
        plt.close('all')