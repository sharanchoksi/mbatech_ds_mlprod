#Imports
import pandas as pd
import config
from sklearn.preprocessing import LabelEncoder
import model_dispatcher
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import os
import joblib
import argparse
import warnings
warnings.filterwarnings('ignore')

#run
def run(fold,model):

    df=pd.read_csv(config.TRAINING_FILE)
    
    # Creating train & validation dataframes
    df_train = df[df.kfold!=fold].reset_index(drop=True)
    df_valid = df[df.kfold==fold].reset_index(drop=True)

    #Preprocessing 
    df_train['Item_Weight'].fillna(df_train['Item_Weight'].mean(),inplace=True)
    df_valid['Item_Weight'].fillna(df_valid['Item_Weight'].mean(),inplace=True)

    df_train['Outlet_Size'].fillna(df_train['Outlet_Size'].mode()[0],inplace=True)
    df_valid['Outlet_Size'].fillna(df_valid['Outlet_Size'].mode()[0],inplace=True)

    #Simple Feature Engineering
    df_train['Outlet_Age']=df_train['Outlet_Establishment_Year'].apply(lambda year:2024-year)
    df_valid['Outlet_Age']=df_valid['Outlet_Establishment_Year'].apply(lambda year:2024-year)

    #Dropping unnecessary cols
    df_train=df_train.drop(['Item_Identifier','Outlet_Identifier','Outlet_Establishment_Year'],axis=1)
    df_valid=df_valid.drop(['Item_Identifier','Outlet_Identifier','Outlet_Establishment_Year'],axis=1)

    #Label Encoding
    variables=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type','Item_Type']
    label_encoders={}

    for vars in variables:
        le=LabelEncoder()
        df_train[vars]=le.fit_transform(df_train[vars])  
        df_valid[vars]=le.transform(df_valid[vars])
        label_encoders[vars] = le

    #train val
    x_train=df_train[['Item_Weight','Item_Fat_Content','Item_Visibility','Item_Type','Item_MRP','Outlet_Size','Outlet_Location_Type','Outlet_Type','Outlet_Age']]
    y_train=df_train['Item_Outlet_Sales']

    x_valid=df_valid[['Item_Weight','Item_Fat_Content','Item_Visibility','Item_Type','Item_MRP','Outlet_Size','Outlet_Location_Type','Outlet_Type','Outlet_Age']]
    y_valid=df_valid['Item_Outlet_Sales']        

    #Model fitting
    reg=model_dispatcher.models[model]

    reg.fit(x_train,y_train)

    pred=reg.predict(x_valid)

    #Print
    mse=mean_squared_error(y_valid,pred)
    print(f'fold={fold},MSE={mse}')
    r2=r2_score(y_valid,pred)
    print(f'fold={fold},R2 Score={r2}')
    mae=mean_absolute_error(y_valid,pred)
    print(f'fold={fold},MAE={mae}')

    #Model file dump
    joblib.dump(
        reg,
        os.path.join(config.MODEL_OUTPUT,f"FOLD={fold} NAME={model}.bin")
    )

    #End

#main

if __name__ == "__main__":
    parser=argparse.ArgumentParser()


    #Arg1
    parser.add_argument(
        '--fold',
        type=int
    )          

    #Arg2
    parser.add_argument(
        '--model',
        type=str
    )

    args= parser.parse_args()
    #some work
    run(
        args.fold,
        args.model
    )