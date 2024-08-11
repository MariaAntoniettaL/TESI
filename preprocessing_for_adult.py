from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



def preprocessing_funct_not_enc(df):

    col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                'marital-status', 'occupation', 'relationship', 'race', 'sex',
                'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
    
    def income_map(income):
        if isinstance(income, str):
            income = income.strip()
            if income == ">50K":
                return 1
        return 0
    df["income"] = df["income"].map(income_map)
    #print((set(df["income"])))

    df.drop_duplicates(inplace=True)
    
    #Divisione in train(60%), validation(20%), holdon(20%), test(20%)
    df_train, df_temp = train_test_split(df, test_size=0.6, shuffle=True, random_state=42, stratify=df["income"])
    df_temp, df_val = train_test_split(df_temp, test_size=0.3333, shuffle=True, random_state=42, stratify=df_temp["income"])
    df_test, df_holdout = train_test_split(df_temp, test_size=0.5, shuffle=True, random_state=42, stratify=df_temp["income"])
    
    
    #age va da 17 a 90
    #print((set(df_train["age"])))
    # Creazione delle fasce di età
    bins = [0, 24, 34, 44, 54, 64, 100]
    labels_age = ['17-24', '25-34', '35-44', '45-54', '55-64', '65-100']

    # Aggiunta della colonna 'age_group' al DataFrame
    df_train['age_group'] = pd.cut(df_train['age'], bins=bins, labels=labels_age, right=False)
    df_test['age_group'] = pd.cut(df_test['age'], bins=bins, labels=labels_age, right=False)
    df_val['age_group'] = pd.cut(df_val['age'], bins=bins, labels=labels_age, right=False)
    df_holdout['age_group'] = pd.cut(df_holdout['age'], bins=bins, labels=labels_age, right=False)


    #eliminazione feature originale
    df_train.drop(columns=['age'], inplace=True)
    df_test.drop(columns=['age'], inplace=True)
    df_val.drop(columns=['age'], inplace=True)
    df_holdout.drop(columns=['age'], inplace=True)

    
    #WORKCLASS {' Self-emp-not-inc', ' Without-pay', ' State-gov', ' ?', ' Private', ' Self-emp-inc', ' Never-worked', ' Local-gov', ' Federal-gov'}
    df_train['workclass'] = df_train['workclass'].str.strip()
    df_test['workclass'] = df_test['workclass'].str.strip()
    df_val['workclass'] = df_val['workclass'].str.strip()
    df_holdout['workclass'] = df_holdout['workclass'].str.strip()
    #Combinazione di cose simili 
    df_train['workclass'] = df_train['workclass'].replace('?', 'Unknown')
    df_test['workclass'] = df_test['workclass'].replace('?', 'Unknown')
    df_val['workclass'] = df_val['workclass'].replace('?', 'Unknown')
    df_holdout['workclass'] = df_holdout['workclass'].replace('?', 'Unknown')

    # Combine "Federal-gov", "State-gov" e "Local-gov" e rename in "Government"
    df_train.loc[df_train["workclass"].isin(["Federal-gov", "State-gov", "Local-gov"]), "workclass"] = "Government"
    df_test.loc[df_test["workclass"].isin(["Federal-gov", "State-gov", "Local-gov"]), "workclass"] = "Government"
    df_val.loc[df_val["workclass"].isin(["Federal-gov", "State-gov", "Local-gov"]), "workclass"] = "Government"
    df_holdout.loc[df_holdout["workclass"].isin(["Federal-gov", "State-gov", "Local-gov"]), "workclass"] = "Government"

    # Combine "Self-emp-inc", "Self-emp-not-inc" e rename in "Self-emp"
    df_train.loc[df_train["workclass"].isin(["Self-emp-inc", "Self-emp-not-inc"]), "workclass"] = "Self-emp"
    df_test.loc[df_test["workclass"].isin(["Self-emp-inc", "Self-emp-not-inc"]), "workclass"] = "Self-emp"
    df_val.loc[df_val["workclass"].isin(["Self-emp-inc", "Self-emp-not-inc"]), "workclass"] = "Self-emp"
    df_holdout.loc[df_holdout["workclass"].isin(["Self-emp-inc", "Self-emp-not-inc"]), "workclass"] = "Self-emp"

    
    #FNLWGHT only to normalize 
    minmax_s = MinMaxScaler()
    minmax_s.fit(df_train[['fnlwgt']])
    df_train['fnlwgt'] = minmax_s.transform(df_train[['fnlwgt']])
    df_test['fnlwgt'] = minmax_s.transform(df_test[['fnlwgt']])
    df_val['fnlwgt'] = minmax_s.transform(df_val[['fnlwgt']])
    df_holdout['fnlwgt'] = minmax_s.transform(df_holdout[['fnlwgt']])
        
    #EDUCATION: combiniamo alcune features ed encoding
    df_train['education'] = df_train['education'].str.strip()
    df_test['education'] = df_test['education'].str.strip()
    df_val['education'] = df_val['education'].str.strip()
    df_holdout['education'] = df_holdout['education'].str.strip()

    list1 = ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', 'HS-grad', 'Some-college', "12th"]
    df_train.loc[df_train["education"].isin(list1), "education"] = "Non Graduated"
    df_test.loc[df_test["education"].isin(list1), "education"] = "Non Graduated"
    df_val.loc[df_val["education"].isin(list1), "education"] = "Non Graduated"
    df_holdout.loc[df_holdout["education"].isin(list1), "education"] = "Non Graduated"

    list2 = ["Assoc-voc", "Assoc-acdm", "Bachelors"]
    df_train.loc[df_train["education"].isin(list2), "education"] = "Bachelor's Degree"
    df_test.loc[df_test["education"].isin(list2), "education"] = "Bachelor's Degree"
    df_val.loc[df_val["education"].isin(list2), "education"] = "Bachelor's Degree"
    df_holdout.loc[df_holdout["education"].isin(list2), "education"] = "Bachelor's Degree"

    list3 = ["Masters", "Prof-school"]
    df_train.loc[df_train["education"].isin(list3), "education"] = "Master's Degree"
    df_test.loc[df_test["education"].isin(list3), "education"] = "Master's Degree"
    df_val.loc[df_val["education"].isin(list3), "education"] = "Master's Degree"
    df_holdout.loc[df_holdout["education"].isin(list3), "education"] = "Master's Degree"

    list4 = ["Doctorate"]
    df_train.loc[df_train["education"].isin(list4), "education"] = "Doctorate Degree"
    df_test.loc[df_test["education"].isin(list4), "education"] = "Doctorate Degree"
    df_val.loc[df_val["education"].isin(list4), "education"] = "Doctorate Degree"
    df_holdout.loc[df_holdout["education"].isin(list4), "education"] = "Doctorate Degree"

    
     #EDUCATION-NUM, discretizzazione, encoding, dizionario label
    #education-num va da 1 a 16
    #print((set(df_train["education-num"])))
    # Creazione delle fasce di istruzione
    bins = [0, 1, 3, 5, 8, 9, 10, 12, 13, 14, 15, 16]
    labels_edu_num = [
        '1 Preschool', 
        '2-3 Elementary School', 
        '4-5 Middle School', 
        '6-8 High School', 
        '9 High School Graduate', 
        '10 College', 
        "11-12 Associate's Degree", 
        "13 Bachelor's Degree", 
        "14 Master's Degree", 
        '15 Professional Degree', 
        '16 Doctorate Degree'
    ]

    # Aggiunta della colonna 'edu_num_group' al DataFrame
    df_train['edu_num_group'] = pd.cut(df_train['education-num'], bins=bins, labels=labels_edu_num, right=True)
    df_test['edu_num_group'] = pd.cut(df_test['education-num'], bins=bins, labels=labels_edu_num, right=True)
    df_val['edu_num_group'] = pd.cut(df_val['education-num'], bins=bins, labels=labels_edu_num, right=True)
    df_holdout['edu_num_group'] = pd.cut(df_holdout['education-num'], bins=bins, labels=labels_edu_num, right=True)
    #eliminazione feature not encoded originale
    df_train.drop(columns=['education-num'], inplace=True)
    df_test.drop(columns=['education-num'], inplace=True)
    df_val.drop(columns=['education-num'], inplace=True)
    df_holdout.drop(columns=['education-num'], inplace=True)

    
    #MARITAL STATUS combination, encoding (Never-married', Married = ' Married-spouse-absent', ' Married-civ-spouse', ' Married-AF-spouse', ' Separated', ' Divorced' and ' Widowed')
    # Combine ' Married-spouse-absent', ' Married-civ-spouse', ' Married-AF-spouse' e rename in "Married"
    
    # Strip whitespaces from 'marital-status'
    df_train['marital-status'] = df_train['marital-status'].str.strip()
    df_test['marital-status'] = df_test['marital-status'].str.strip()
    df_val['marital-status'] = df_val['marital-status'].str.strip()
    df_holdout['marital-status'] = df_holdout['marital-status'].str.strip()
    
    df_train.loc[df_train["marital-status"].isin(['Married-spouse-absent', 'Married-civ-spouse','Married-AF-spouse']), "marital-status"] = "Married"
    df_test.loc[df_test["marital-status"].isin(['Married-spouse-absent', 'Married-civ-spouse', 'Married-AF-spouse']), "marital-status"] = "Married"
    df_val.loc[df_val["marital-status"].isin(['Married-spouse-absent', 'Married-civ-spouse', 'Married-AF-spouse']), "marital-status"] = "Married"
    df_holdout.loc[df_holdout["marital-status"].isin(['Married-spouse-absent', 'Married-civ-spouse', 'Married-AF-spouse']), "marital-status"] = "Married"

    
    
    
    #OCCUPATION 
    # Strip whitespaces from 'occupation'
    df_train['occupation'] = df_train['occupation'].str.strip()
    df_test['occupation'] = df_test['occupation'].str.strip()
    df_val['occupation'] = df_val['occupation'].str.strip()
    df_holdout['occupation'] = df_holdout['occupation'].str.strip()

    # Rename "?" to "Unknown" in "occupation"
    df_train['occupation'] = df_train['occupation'].replace('?', 'Unknown')
    df_test['occupation'] = df_test['occupation'].replace('?', 'Unknown')
    df_val['occupation'] = df_val['occupation'].replace('?', 'Unknown')
    df_holdout['occupation'] = df_holdout['occupation'].replace('?', 'Unknown')

    # Government-occ
    list1 = ['Armed-Forces', 'Protective-serv', 'Adm-clerical', 'Transport-moving']
    df_train.loc[df_train["occupation"].isin(list1), "occupation"] = "Government-occ"
    df_test.loc[df_test["occupation"].isin(list1), "occupation"] = "Government-occ"
    df_val.loc[df_val["occupation"].isin(list1), "occupation"] = "Government-occ"
    df_holdout.loc[df_holdout["occupation"].isin(list1), "occupation"] = "Government-occ"

    # Private-occ
    list2 = ['Exec-managerial', 'Priv-house-serv', 'Handlers-cleaners', 'Sales']
    df_train.loc[df_train["occupation"].isin(list2), "occupation"] = "Private-occ"
    df_test.loc[df_test["occupation"].isin(list2), "occupation"] = "Private-occ"
    df_val.loc[df_val["occupation"].isin(list2), "occupation"] = "Private-occ"
    df_holdout.loc[df_holdout["occupation"].isin(list2), "occupation"] = "Private-occ"

    # Self-emp-occ
    list3 = ['Farming-fishing', 'Machine-op-inspct', 'Tech-support', 'Craft-repair']
    df_train.loc[df_train["occupation"].isin(list3), "occupation"] = "Self-emp-occ"
    df_test.loc[df_test["occupation"].isin(list3), "occupation"] = "Self-emp-occ"
    df_val.loc[df_val["occupation"].isin(list3), "occupation"] = "Self-emp-occ"
    df_holdout.loc[df_holdout["occupation"].isin(list3), "occupation"] = "Self-emp-occ"

    #RELATIONSHIP to encode
    #RACE to encode 
    #SEX to encode 
   


    #CAPITAL GAIN E CAPITAL LOSS NORMALIZZO 
    #capital gain
    minmax_s.fit(df_train[['capital-gain']])
    df_train['capital-gain'] = minmax_s.transform(df_train[['capital-gain']])
    df_test['capital-gain'] = minmax_s.transform(df_test[['capital-gain']])
    df_val['capital-gain'] = minmax_s.transform(df_val[['capital-gain']])
    df_holdout['capital-gain'] = minmax_s.transform(df_holdout[['capital-gain']])

    #capital loss
    minmax_s.fit(df_train[['capital-loss']])
    df_train['capital-loss'] = minmax_s.transform(df_train[['capital-loss']])
    df_test['capital-loss'] = minmax_s.transform(df_test[['capital-loss']])
    df_val['capital-loss'] = minmax_s.transform(df_val[['capital-loss']])
    df_holdout['capital-loss'] = minmax_s.transform(df_holdout[['capital-loss']])

    #HOURS-PER-WEEK discretizzazione, encoding 
    bins = [0, 35, 40, 100]
    labels_hours_per_week = ['Part-time', 'Full-time', 'Overtime']

    # Applicare la discretizzazione ai DataFrame
    df_train['hours_per_week_group'] = pd.cut(df_train['hours-per-week'], bins=bins, labels=labels_hours_per_week, right=False)
    df_test['hours_per_week_group'] = pd.cut(df_test['hours-per-week'], bins=bins, labels=labels_hours_per_week, right=False)
    df_val['hours_per_week_group'] = pd.cut(df_val['hours-per-week'], bins=bins, labels=labels_hours_per_week, right=False)
    df_holdout['hours_per_week_group'] = pd.cut(df_holdout['hours-per-week'], bins=bins, labels=labels_hours_per_week, right=False)

    #eliminazione feature originale
    df_train.drop(columns=['hours-per-week'], inplace=True)
    df_test.drop(columns=['hours-per-week'], inplace=True)
    df_val.drop(columns=['hours-per-week'], inplace=True)
    df_holdout.drop(columns=['hours-per-week'], inplace=True)

    #NATIVE COUNTRY combinations, encoding
    df_train['native-country'] = df_train['native-country'].str.strip()
    df_test['native-country'] = df_test['native-country'].str.strip()
    df_val['native-country'] = df_val['native-country'].str.strip()
    df_holdout['native-country'] = df_holdout['native-country'].str.strip()

    # Rename "?" con "Unknown" in "native-country"
    df_train['native-country'] = df_train['native-country'].replace('?', 'Unknown')
    df_test['native-country'] = df_test['native-country'].replace('?', 'Unknown')
    df_val['native-country'] = df_val['native-country'].replace('?', 'Unknown')
    df_holdout['native-country'] = df_holdout['native-country'].replace('?', 'Unknown')

    #caucasian/white
    list1 = ['Germany', 'England', 'Scotland', 'France', 'Italy', 'Ireland', 'Greece', 'Poland', 'Portugal', 'Yugoslavia', 'Hungary']

    df_train.loc[df_train["native-country"].isin(list1), "native-country"] = " Caucasian-White"
    df_test.loc[df_test["native-country"].isin(list1), "native-country"] = " Caucasian-White"
    df_val.loc[df_val["native-country"].isin(list1), "native-country"] = " Caucasian-White"
    df_holdout.loc[df_holdout["native-country"].isin(list1), "native-country"] = " Caucasian-White"

    #african american/black
    list2 = ['United-States', 'Jamaica', 'Haiti', 'Trinadad&Tobago']
    df_train.loc[df_train["native-country"].isin(list2), "native-country"] = " African-American-Black"
    df_test.loc[df_test["native-country"].isin(list2), "native-country"] = " African-American-Black"
    df_val.loc[df_val["native-country"].isin(list2), "native-country"] = " African-American-Black"
    df_holdout.loc[df_holdout["native-country"].isin(list2), "native-country"] = " African-American-Black"

    #latino/hispanic
    list3 = ['Peru', 'Mexico', 'Puerto-Rico', 'Guatemala', 'Honduras', 'El-Salvador', 'Nicaragua', 'Cuba', 'Dominican-Republic', 'Columbia', 'Ecuador']
    df_train.loc[df_train["native-country"].isin(list3), "native-country"] = " Latino-Hispanic"
    df_test.loc[df_test["native-country"].isin(list3), "native-country"] = " Latino-Hispanic"
    df_val.loc[df_val["native-country"].isin(list3), "native-country"] = " Latino-Hispanic"
    df_holdout.loc[df_holdout["native-country"].isin(list3), "native-country"] = " Latino-Hispanic"

    #Asian
    list4 = ['Thailand', 'Philippines', 'Vietnam', 'China', 'Japan', 'India', 'Taiwan', 'Cambodia', 'Laos' ]
    df_train.loc[df_train["native-country"].isin(list4), "native-country"] = " Asian"
    df_test.loc[df_test["native-country"].isin(list4), "native-country"] = " Asian"
    df_val.loc[df_val["native-country"].isin(list4), "native-country"] = " Asian"
    df_holdout.loc[df_holdout["native-country"].isin(list4), "native-country"] = " Asian"

    #Other
    list5 = ['Outlying-US(Guam-USVI-etc)', 'Iran', 'Unknown', 'Canada', 'South', 'Hong', 'Israel', 'Lebanon', 'Holand-Netherlands', 'Romania', 'Russia', 'Switzerland', 'Scotland'  ]
    df_train.loc[df_train["native-country"].isin(list5), "native-country"] = " Other"
    df_test.loc[df_test["native-country"].isin(list5), "native-country"] = " Other"
    df_val.loc[df_val["native-country"].isin(list5), "native-country"] = " Other"
    df_holdout.loc[df_holdout["native-country"].isin(list5), "native-country"] = " Other"
    
        
    return df_train, df_test, df_val, df_holdout


'''def encoding_funct(df_train, df_test, df_val, df_holdout, categorical_cols):
    df_train_enc = df_train.copy()
    df_test_enc = df_test.copy()
    df_val_enc = df_val.copy()
    df_holdout_enc = df_holdout.copy()
    # Initialize dictionaries
    categorical_names = {}
    le_dict = {}

    # Ensure categorical_cols contains valid indices
    for feature in categorical_cols:
        if feature < len(df_train_enc.columns):  # Ensure the index is within the column range
            le = LabelEncoder()
            transformed_values = le.fit_transform(df_train_enc.iloc[:, feature])
            df_train_enc.iloc[:, feature] = transformed_values
            categorical_names[feature] = le.classes_
            le_dict[feature] = le
        else:
            print(f"Index {feature} is out of bounds for columns in df_train_enc.")

    # Print out the encoding details
    for feature, classes in categorical_names.items():
        print(f'Feature index {feature}: ')
        print('Encoded values: ', list(range(len(classes))))
        print('Original values: ', classes)
        print()

    # Handling non-categorical columns
    non_categorical_cols = [i for i in range(len(df_train_enc.columns)) if i not in categorical_cols]

    for col in non_categorical_cols:
        le = LabelEncoder()
        transformed_values = le.fit_transform(df_train_enc.iloc[:, col])
        df_train_enc.iloc[:, col] = transformed_values
        
    return df_train_enc, df_test_enc, df_val_enc, df_holdout_enc
    '''

def get_original_label(column_name, encoded_value):
        original_labels_dict = {
        'workclass': ['Government', 'Never-worked', 'Private', 'Self-emp', 'Unknown', 'Without-pay'],
        'education': ["Bachelor's Degree", 'Doctorate Degree', "Master's Degree", 'Non Graduated'],
        'marital-status': ['Divorced', 'Married', 'Never-married', 'Separated', 'Widowed'],
        'occupation': ['Self-emp-occ', 'Government-occ', 'Other-service', 'Private-occ', 'Prof-specialty', 'Unknown'],
        'relationship': ['Husband', 'Not-in-family', 'Other-relative', 'Own-child', 'Unmarried', 'Wife'],
        'race': ['Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White'],
        'sex': ['Female', 'Male'],
        'native-country': ['Other', 'African-American-Black', 'Asian', 'Caucasian-White', 'Latino-Hispanic']}
        return original_labels_dict[column_name][encoded_value]