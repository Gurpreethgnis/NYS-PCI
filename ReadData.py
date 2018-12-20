
class Read_Data():
    
    def __init__(self):
        
        import pandas as pd
        import numpy as np
        pd.options.mode.chained_assignment=None
        import matplotlib.pyplot as plt

    def read_dataset(self,inputdir):
        import pandas as pd
        #load dataset
        data=pd.read_csv(inputdir,low_memory=False,header=None,keep_default_na=False)
        return data
    
    def load_data(self,infile=[],snxdata=[],snydata=[],sncatcols=[],snconcols=[],indir=[]):
        import pandas as pd
        x_savename=indir+'/Data/'+snxdata
        y_savename=indir+'/Data/'+snydata
        catcols_savename=indir+'/Data/'+sncatcols
        concols_savename=indir+'/Data/'+snconcols

        try:
            x_data=pd.read_csv(x_savename)
            y_data=pd.read_csv(y_savename)
            cat_cols=pd.read_csv(catcols_savename)
            con_cols=pd.read_csv(concols_savename)
            Data={}
            Data['x_data']=x_data
            Data['y_data']=y_data
            Data['con_cols']=con_cols
            Data['cat_cols']=cat_cols
        except:
            print "-"*50
            print "Could not find save CSVs. Creatind new ones."
            data_raw=self.read_dataset(inputdir=indir+'/Data/'+infile)
            Data=self.process_dataset(data_raw)
            Data['x_data'].to_csv(x_savename,index=False)
            Data['y_data'].to_csv(y_savename,index=False)
            pd.DataFrame({'con_cols':Data['con_cols']}).to_csv(catcols_savename,index=False)
            pd.DataFrame({'cat_cols':Data['cat_cols']}).to_csv(concols_savename,index=False)

        return Data

    def process_dataset(self,data,keep_cols_idx=0,cat_con_cols_idx=1,colnames_idx=2,raw_data_start_col_idx=3):
        import pandas as pd
        import numpy as np
        print "-------------------------------------------------------------------------------"
        print "Starting processing data\n"

        print "Assuming keep_cols are located in row "+str(keep_cols_idx)+"\n"
        keep_cols=data.iloc[keep_cols_idx]
        print "Assuming indormation about categorical columns is located in row "+str(cat_con_cols_idx)+"\n"
        cat_con_cols=data.iloc[cat_con_cols_idx]
        print "Assuming column names are located in row "+str(colnames_idx)+"\n"
        colnames=data.iloc[colnames_idx]
        print "Assuming Raw Data is from "+str(raw_data_start_col_idx)+' to '+str(raw_data_start_col_idx)+"\n"
        raw_data=data.iloc[raw_data_start_col_idx:,:]

        #get list of columns to keep, remove and output variable

        kcol_idx=[]
        rcol_idx=[]
        ocol_idx=[]
        for colnum in range(0,len(keep_cols)):
            if int(colnum) !=0 and not pd.isnull(keep_cols[colnum]):
                if int(keep_cols[colnum])==1:
                    kcol_idx.append(colnum)
                elif int(keep_cols[colnum])==999:
                    ocol_idx.append(colnum)
                else:
                    rcol_idx.append(colnum)  
            else:
                    rcol_idx.append(colnum)  
        print "The number of columns in the dataset should be "+str(len(kcol_idx))


        #reslice the data and check if the columns match

        y_data=raw_data.iloc[:,ocol_idx]
        x_data=raw_data.iloc[:,kcol_idx]

        if (len(kcol_idx))==(x_data.shape[1]):
            print "Col numbers match"
        else:
            print "COL NUMBERS DONT MATCH!" 

        #Get the list of categorical and continous columns

        All_cols=cat_con_cols[kcol_idx]

        cat_cols=[]
        con_cols=[]

        for colnum in All_cols.index:
                if int(All_cols[colnum])==1:
                    cat_cols.append(colnames[colnum])
                else:
                    con_cols.append(colnames[colnum])

        #renaming columns in the dataset

        n_colname=colnames[kcol_idx]
        x_data.rename(columns=n_colname,inplace=True)
        x_data[con_cols]=x_data[con_cols].apply(pd.to_numeric,errors='coerce')
        x_data[cat_cols]=x_data[cat_cols].replace(' ',np.nan)        
        y_data=y_data.astype('int')
        y_data.columns=y_data.columns.astype(str)
        y_data.rename(columns={"100":'Label'},inplace=True)
        y_data['Label'].value_counts()
        print "Successfully processed the dataset"
        print "-------------------------------------------------------------------------------"
        print "#Information#"
        print "Use (return_variable).keys() to access the processed files\n"
        print "The class has a function to remove variables with specified missing percentage"
        print "Usage for find_missing_data() is as follow"
        print "find_missing_data(x_data=x_data_file,missing_percentage=some_numerical_percentage,cat_cols=all_categorical_cols,con_cols=all_continous_cols)"

        return {'x_data':x_data,'y_data':y_data,'cat_cols':cat_cols,'con_cols':con_cols }
    
    def filter_missing_data(self,x_data=[],missing_percentage=80,cat_cols=[],con_cols=[]):
        import numpy as np
        #check missing data
        missing_df=x_data.isnull().sum(axis=0).reset_index()
        missing_df.columns=['column_name','missing_count']
        missing_df=missing_df.loc[missing_df['missing_count']>0]
        missing_df=missing_df.sort_values(by='missing_count')
        missing_df['missing_count']=(missing_df['missing_count'].astype('float32')/x_data.shape[0])*100

        #removing the variables with missing data > missing percentage

        filter_missing_percentage=np.int(missing_percentage)
        cols_to_exclude=missing_df['column_name'][missing_df['missing_count']>=filter_missing_percentage].tolist()
        print "Total variables in the dataset : "+str(x_data.shape[1])
        print "Cols with missing data "+str(filter_missing_percentage)+" % : "+str(len(cols_to_exclude))
        print "Number of variables in the new dataset : "+str(x_data.shape[1]-len(cols_to_exclude))
        x_filtered=x_data.drop(cols_to_exclude,axis=1)
        x_filtered_w_missing=x_filtered.copy()

        #removing the variables names with missing data > missing percentage
        
        new_con_cols=[]
        rem_con_cols=[]
        for i in range(0,len(con_cols.con_cols)):
            colnames=con_cols.con_cols[i] 
            if colnames in x_filtered.columns:
                new_con_cols.append(colnames)
            else:
                rem_con_cols.append(colnames)

        new_cat_cols=[]
        rem_cat_cols=[]
        for i in range(0,len(cat_cols.cat_cols)):
            colnames=cat_cols.cat_cols[i] 
            if colnames in x_filtered.columns:
                new_cat_cols.append(colnames)  
            else:
                rem_cat_cols.append(colnames)
        print "Reduced number of continous variables from "+str(len(con_cols))+" to "+str(len(new_con_cols))
        print "Reduced number of categorical variables from "+str(len(cat_cols))+" to "+str(len(new_cat_cols))
        print "-------------------------------------------------------------------------------"
        return {'x_data':x_filtered,'con_cols':new_con_cols,'cat_cols':new_cat_cols,'rem_cat_cols':rem_cat_cols,'rem_con_cols':rem_con_cols}
