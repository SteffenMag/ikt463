import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

def categorize_col(col, col_name):
    quantiles = pd.qcut(col, q=3, duplicates="drop")
    unique_bins = quantiles.cat.categories.size  # Number of unique bins after dropping duplicates
    labels = [f"{col_name}_low", f"{col_name}_medium", f"{col_name}_high"][:unique_bins]

    quantiles = pd.qcut(col, q=unique_bins, labels=labels)
    return pd.get_dummies(quantiles)

def load_data(upscale=True,downscale=False,save=True,num_classes=3):
    df = pd.read_csv('data/data.csv',delimiter=';')\
    # Replace symbols in column names with underlines
    df.columns = df.columns.str.replace('(', '')
    df.columns = df.columns.str.replace(')', '')
    df.columns = df.columns.str.replace('/', ' ')
    df.columns = df.columns.str.replace("'", '')
    df.columns = df.columns.str.replace(' ', '_')
    
    # drop these columns 
    print('Dropping these columns: \n - Marital_status \n - Application_order \n - Course \n - Nacionality \n '
          '- Previous_qualification \n - Previous_qualification_grade \n - Curricular_units_1st_sem_credited \n '
          '- Curricular_units_1st_sem_evaluations \n - Unemployment_rate \n - Inflation_rate \n - GDP \n - Curricular_units_1st_sem_without_evaluations \n '
          '- Curricular_units_2nd_sem_credited \n - Curricular_units_2nd_sem_without_evaluations \n - Mothers_qualification \n - Fathers_qualification \n '
          '- Mothers_occupation \n - Fathers_occupation \n - Admission_grade \n - Gender \n - Daytime_evening_attendance \n - Educational_special_needs \n - International')
    
    
    df = df.drop(['Marital_status','Application_order','Course','Nacionality','Previous_qualification','Previous_qualification_grade',
                  'Curricular_units_1st_sem_credited','Curricular_units_1st_sem_evaluations','Unemployment_rate', 
                  'Inflation_rate', 'GDP','Curricular_units_1st_sem_without_evaluations','Curricular_units_2nd_sem_credited',
                  'Curricular_units_2nd_sem_without_evaluations','Mothers_qualification', 'Fathers_qualification', 
                  'Mothers_occupation', 'Fathers_occupation', 'Admission_grade', 'Gender','Daytime_evening_attendance','Educational_special_needs','International'], axis=1)

    
    df_copy = df.copy()

    categorize_cols = ['Curricular_units_1st_sem_grade', 'Curricular_units_1st_sem_approved', 'Curricular_units_2nd_sem_grade', 'Curricular_units_2nd_sem_approved',
                        'Curricular_units_2nd_sem_evaluations','Age_at_enrollment','Application_mode','Curricular_units_1st_sem_enrolled','Curricular_units_2nd_sem_enrolled']
    '''
    # keep 15 highest cols + some custom changes
    df = df.drop(['Marital_status','Course','Nacionality','Previous_qualification',
                  'Curricular_units_1st_sem_credited','Unemployment_rate', 
                  'Inflation_rate', 'GDP','Curricular_units_1st_sem_without_evaluations','Curricular_units_2nd_sem_credited',
                  'Curricular_units_2nd_sem_without_evaluations','Mothers_qualification', 'Fathers_qualification', 
                  'Mothers_occupation', 'Fathers_occupation', 'Gender','Daytime_evening_attendance','Educational_special_needs','International'], axis=1)
    
    
    
    df_copy = df.copy()

    categorize_cols = ['Curricular_units_1st_sem_grade', 'Curricular_units_1st_sem_approved', 'Curricular_units_2nd_sem_grade', 'Curricular_units_2nd_sem_approved',
                        'Curricular_units_2nd_sem_evaluations','Age_at_enrollment','Application_mode','Curricular_units_1st_sem_enrolled','Curricular_units_2nd_sem_enrolled',
                      'Curricular_units_1st_sem_evaluations','Admission_grade','Previous_qualification_grade','Application_order']'''
    # last row is the added columns
    
    binarize_cols = []

    for col in categorize_cols:
        column = df_copy[f'{col}']

        # make a new column that is called 'missing_col_name' that is 1 if the value is missing and 0 otherwise
        df_copy[f'missing_{col}'] = column.isnull().astype(int)
        categorize_cols = categorize_col(column, col)
        df_copy = pd.concat([df_copy, categorize_cols], axis=1)
        df_copy = df_copy.drop([col], axis=1)

    for col in binarize_cols:
        df_copy[col] = df_copy[col].apply(lambda x: 1 if x == 0 else 0)

    #one_hot_encoded_data = pd.get_dummies(df_copy, columns=['Marital_status','Application_order'], dtype=int)
    one_hot_encoded_data = df_copy
    print(one_hot_encoded_data.columns)    

    # remove the rows with target value Enrolled if num_classes 2
    if num_classes==2:
        one_hot_encoded_data = one_hot_encoded_data[one_hot_encoded_data['Target'] != 'Enrolled']
    print(one_hot_encoded_data.columns)
    X, y = one_hot_encoded_data.drop('Target', axis=1), one_hot_encoded_data['Target']
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    if upscale:
        smote = SMOTE(random_state=42)
        train_x, train_y = smote.fit_resample(train_x, train_y)
        test_x, test_y = smote.fit_resample(test_x, test_y)
      
    if downscale:
        rus = RandomUnderSampler(random_state=42, sampling_strategy = 'auto')
        train_x, train_y = rus.fit_resample(train_x, train_y)
        test_x, test_y = rus.fit_resample(test_x, test_y)
    
    # change all y values to 0, 1 or 2, 0 if the target is dropout, 1 if the target is Graduate, 2 if the target is enrolled
    #pd.set_option("future.no_silent_downcasting", True)
    
    train_y = train_y.replace('Dropout', 0)
    train_y = train_y.replace('Graduate', 1)
    train_y = train_y.replace('Enrolled', 2)
    
    print(train_y.value_counts())
    
    test_y = test_y.replace('Dropout', 0)
    test_y = test_y.replace('Graduate', 1)
    test_y = test_y.replace('Enrolled', 2)
    
    print(test_y.value_counts())
    
    
    train_y = train_y.to_numpy(dtype=np.uint32)
    test_y = test_y.to_numpy(dtype=np.uint32)
    
    
    # change x from df to numpy array
    train_x = train_x.to_numpy(dtype=np.uint8)
    test_x = test_x.to_numpy(dtype=np.uint8)
    
    # save the train and test data
    if save:
        np.save('tm_data/train_x.npy', train_x)
        np.save('tm_data/train_y.npy', train_y)
        np.save('tm_data/test_x.npy', test_x)
        np.save('tm_data/test_y.npy', test_y)
    
    return train_x, train_y, test_x, test_y

def load_orig_data(upscale=False,downscale=False,save=True,num_classes=3):
    df = pd.read_csv('data/data.csv',delimiter=';')\
    #df = pd.read_csv('data/train.csv',delimiter=',')\
    
    # Replace symbols in column names with underlines
    df.columns = df.columns.str.replace('(', '')
    df.columns = df.columns.str.replace(')', '')
    df.columns = df.columns.str.replace('/', ' ')
    df.columns = df.columns.str.replace("'", '')
    df.columns = df.columns.str.replace(' ', '_')

    df_copy = df.copy()
    #print(df_copy.columns)
    df = df.drop(['Gender','International','Nacionality','Mothers_qualification', 'Fathers_qualification', 
                  'Mothers_occupation', 'Fathers_occupation'],axis=1)

    categorize_cols = ['Curricular_units_1st_sem_grade', 'Curricular_units_1st_sem_approved', 'Curricular_units_1st_sem_enrolled', 
                    'Curricular_units_1st_sem_evaluations','Curricular_units_2nd_sem_grade', 'Curricular_units_2nd_sem_approved', 
                       'Curricular_units_2nd_sem_enrolled', 'Curricular_units_2nd_sem_evaluations','Previous_qualification_grade','Age_at_enrollment', 
                       'Admission_grade','Unemployment_rate', 'Inflation_rate', 'GDP']

    binarize_cols = ['Curricular_units_1st_sem_credited','Curricular_units_1st_sem_without_evaluations', 'Curricular_units_2nd_sem_credited', 
                    'Curricular_units_2nd_sem_without_evaluations', 'Previous_qualification']

    for col in categorize_cols:
        column = df_copy[f'{col}']

        # make a new column that is called 'missing_col_name' that is 1 if the value is missing and 0 otherwise
        df_copy[f'missing_{col}'] = column.isnull().astype(int)
        categorize_cols = categorize_col(column, col)
        df_copy = pd.concat([df_copy, categorize_cols], axis=1)
        df_copy = df_copy.drop([col], axis=1)

    for col in binarize_cols:
        df_copy[col] = df_copy[col].apply(lambda x: 1 if x == 0 else 0)

    #one_hot_encoded_data = pd.get_dummies(df_copy, columns=['Marital_status','Application_mode','Application_order','Course','Nacionality',
    #                                                        'Mothers_qualification', 'Fathers_qualification', 'Mothers_occupation', 
    #                                                        'Fathers_occupation'], dtype=int)
    
    one_hot_encoded_data = pd.get_dummies(df_copy, columns=['Marital_status','Application_mode','Application_order','Course'], dtype=int)
    
    #one_hot_encoded_data = pd.read_csv('data/processed_data.csv')
    if num_classes==2:
        one_hot_encoded_data = one_hot_encoded_data[one_hot_encoded_data['Target'] != 'Enrolled']
    print(one_hot_encoded_data.columns)
    
    X, y = one_hot_encoded_data.drop('Target', axis=1), one_hot_encoded_data['Target']
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    if upscale:
        smote = SMOTE(random_state=42)
        train_x, train_y = smote.fit_resample(train_x, train_y)
        test_x, test_y = smote.fit_resample(test_x, test_y)
      
    if downscale:
        rus = RandomUnderSampler(random_state=42, sampling_strategy = 'auto')
        train_x, train_y = rus.fit_resample(train_x, train_y)
        test_x, test_y = rus.fit_resample(test_x, test_y)
    
    # change all y values to 0, 1 or 2, 0 if the target is dropout, 1 if the target is Graduate, 2 if the target is enrolled
    #pd.set_option("future.no_silent_downcasting", True)
    
    train_y = train_y.replace('Dropout', 0)
    train_y = train_y.replace('Graduate', 1)
    train_y = train_y.replace('Enrolled', 2)
    
    print(train_y.value_counts())
    
    test_y = test_y.replace('Dropout', 0)
    test_y = test_y.replace('Graduate', 1)
    test_y = test_y.replace('Enrolled', 2)
    
    print(test_y.value_counts())
    
    
    train_y = train_y.to_numpy(dtype=np.uint32)
    test_y = test_y.to_numpy(dtype=np.uint32)
    
    
    # change x from df to numpy array
    train_x = train_x.to_numpy(dtype=np.uint8)
    test_x = test_x.to_numpy(dtype=np.uint8)
    
    # save the train and test data
    if save:
        np.save('tm_data/train_x.npy', train_x)
        np.save('tm_data/train_y.npy', train_y)
        np.save('tm_data/test_x.npy', test_x)
        np.save('tm_data/test_y.npy', test_y)
    
    return train_x, train_y, test_x, test_y


