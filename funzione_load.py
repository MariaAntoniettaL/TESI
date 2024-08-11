from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#Global mappings
age_mapping = {}
inverse_age_mapping = {}

workclass_mapping = {}
inverse_workclass_mapping = {}

education_mapping = {}
inverse_education_mapping = {}

education_num_mapping = {}
inverse_education_num_mapping = {}

marital_status_mapping = {}
inverse_marital_status_mapping = {}

occupation_mapping ={}
inverse_occupation_mapping = {}

relationship_mapping = {}
inverse_relationship_mapping = {}

race_mapping = {}
inverse_race_mapping = {}

sex_mapping = {}
inverse_sex_mapping = {}

hours_per_week_mapping = {}
inverse_hours_per_week_mapping = {}

native_country_mapping = {}
inverse_native_country_mapping = {}

def preprocessing_funct(df):
    global age_mapping, workclass_mapping, education_mapping, education_mapping, education_num_mapping, marital_status_mapping, occupation_mapping, relationship_mapping, race_mapping, sex_mapping, hours_per_week_mapping, native_country_mapping
    global inverse_age_mapping, inverse_workclass_mapping, inverse_education_mapping, inverse_education_mapping, inverse_education_num_mapping, inverse_marital_status_mapping, inverse_occupation_mapping, inverse_relationship_mapping, inverse_race_mapping, inverse_sex_mapping, inverse_hours_per_week_mapping, inverse_native_country_mapping
    
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
    bins = [17, 24, 34, 44, 54, 64, 90]
    labels_age = ['17-24', '25-34', '35-44', '45-54', '55-64', '65-100']

    # Aggiunta della colonna 'age_group' al DataFrame
    df_train['age_group'] = pd.cut(df_train['age'], bins=bins, labels=labels_age, right=True)
    df_test['age_group'] = pd.cut(df_test['age'], bins=bins, labels=labels_age, right=True)
    df_val['age_group'] = pd.cut(df_val['age'], bins=bins, labels=labels_age, right=True)
    df_holdout['age_group'] = pd.cut(df_holdout['age'], bins=bins, labels=labels_age, right=True)

    #eliminazione feature originale
    df_train.drop(columns=['age'], inplace=True)
    df_test.drop(columns=['age'], inplace=True)
    df_val.drop(columns=['age'], inplace=True)
    df_holdout.drop(columns=['age'], inplace=True)

    #encoding e dizionario 
    age_mapping = {label: idx for idx, label in enumerate(labels_age)}
    inverse_age_mapping = {idx: label for label, idx in age_mapping.items()}
    
    df_train['age_group_enc'] = df_train['age_group'].apply(lambda x: age_mapping[x])
    df_test['age_group_enc'] = df_test['age_group'].apply(lambda x: age_mapping[x])
    df_val['age_group_enc'] = df_val['age_group'].apply(lambda x: age_mapping[x])
    df_holdout['age_group_enc'] = df_holdout['age_group'].apply(lambda x: age_mapping[x])
    
    df_train.drop(columns=['age_group'], inplace=True)
    df_test.drop(columns=['age_group'], inplace=True)
    df_val.drop(columns=['age_group'], inplace=True)
    df_holdout.drop(columns=['age_group'], inplace=True)
    
    '''#WORKCLASS {' Self-emp-not-inc', ' Without-pay', ' State-gov', ' ?', ' Private', ' Self-emp-inc', ' Never-worked', ' Local-gov', ' Federal-gov'}
    df['workclass'] = df_train['workclass'].str.strip()
    df_train['workclass'] = df_train['workclass'].str.strip()
    df_test['workclass'] = df_test['workclass'].str.strip()
    df_val['workclass'] = df_val['workclass'].str.strip()
    df_holdout['workclass'] = df_holdout['workclass'].str.strip()
    #Combinazione di cose simili 
    df['workclass'] = df['workclass'].replace('?', 'Unknown')
    df_train['workclass'] = df_train['workclass'].replace('?', 'Unknown')
    df_test['workclass'] = df_test['workclass'].replace('?', 'Unknown')
    df_val['workclass'] = df_val['workclass'].replace('?', 'Unknown')
    df_holdout['workclass'] = df_holdout['workclass'].replace('?', 'Unknown')

    # Combine "Federal-gov", "State-gov" e "Local-gov" e rename in "Government"
    df.loc[df["workclass"].isin(["Federal-gov", "State-gov", "Local-gov"]), "workclass"] = "Government"
    df_train.loc[df_train["workclass"].isin(["Federal-gov", "State-gov", "Local-gov"]), "workclass"] = "Government"
    df_test.loc[df_test["workclass"].isin(["Federal-gov", "State-gov", "Local-gov"]), "workclass"] = "Government"
    df_val.loc[df_val["workclass"].isin(["Federal-gov", "State-gov", "Local-gov"]), "workclass"] = "Government"
    df_holdout.loc[df_holdout["workclass"].isin(["Federal-gov", "State-gov", "Local-gov"]), "workclass"] = "Government"

    # Combine "Self-emp-inc", "Self-emp-not-inc" e rename in "Self-emp"
    df.loc[df["workclass"].isin(["Self-emp-inc", "Self-emp-not-inc"]), "workclass"] = "Self-emp"
    df_train.loc[df_train["workclass"].isin(["Self-emp-inc", "Self-emp-not-inc"]), "workclass"] = "Self-emp"
    df_test.loc[df_test["workclass"].isin(["Self-emp-inc", "Self-emp-not-inc"]), "workclass"] = "Self-emp"
    df_val.loc[df_val["workclass"].isin(["Self-emp-inc", "Self-emp-not-inc"]), "workclass"] = "Self-emp"
    df_holdout.loc[df_holdout["workclass"].isin(["Self-emp-inc", "Self-emp-not-inc"]), "workclass"] = "Self-emp"


    #encoding e dizionario 
    # Creazione del dizionario per il mapping delle classi di lavoro
    unique_workclasses = df['workclass'].unique()
    
    workclass_mapping = {label: idx for idx, label in enumerate(unique_workclasses)}
    inverse_workclass_mapping = {idx: label for label, idx in workclass_mapping.items()}
    #mapping
    df_train['workclass_enc'] = df_train['workclass'].apply(lambda x: workclass_mapping[x])
    df_test['workclass_enc'] = df_test['workclass'].apply(lambda x: workclass_mapping[x])
    df_val['workclass_enc'] = df_val['workclass'].apply(lambda x: workclass_mapping[x])
    df_holdout['workclass_enc'] = df_holdout['workclass'].apply(lambda x: workclass_mapping[x])

    # Eliminazione della feature non encoded
    df_train.drop(columns=['workclass'], inplace=True)
    df_test.drop(columns=['workclass'], inplace=True)
    df_val.drop(columns=['workclass'], inplace=True)
    df_holdout.drop(columns=['workclass'], inplace=True)


    # Visualizzazione del risultato
    #print("Train DataFrame after encoding:")
    #print(df_train[['workclass_enc']].head())

    #print("Workclass mapping:")
    #print(workclass_mapping)

    #print("Inverse workclass mapping:")
    #print(inverse_workclass_mapping)'''
    
        # Pulizia e normalizzazione dei dati
    df['workclass'] = df['workclass'].str.strip()
    df_train['workclass'] = df_train['workclass'].str.strip()
    df_test['workclass'] = df_test['workclass'].str.strip()
    df_val['workclass'] = df_val['workclass'].str.strip()
    df_holdout['workclass'] = df_holdout['workclass'].str.strip()

    # Sostituzione dei valori sconosciuti
    df['workclass'] = df['workclass'].replace('?', 'Unknown')
    df_train['workclass'] = df_train['workclass'].replace('?', 'Unknown')
    df_test['workclass'] = df_test['workclass'].replace('?', 'Unknown')
    df_val['workclass'] = df_val['workclass'].replace('?', 'Unknown')
    df_holdout['workclass'] = df_holdout['workclass'].replace('?', 'Unknown')

    # Combinazione delle categorie
    df['workclass'] = df['workclass'].replace({
        'Federal-gov': 'Government', 
        'State-gov': 'Government', 
        'Local-gov': 'Government',
        'Self-emp-inc': 'Self-emp', 
    'Self-emp-not-inc': 'Self-emp'
    })
    df_train['workclass'] = df_train['workclass'].replace({
        'Federal-gov': 'Government', 
        'State-gov': 'Government', 
        'Local-gov': 'Government',
        'Self-emp-inc': 'Self-emp', 
        'Self-emp-not-inc': 'Self-emp'
    })
    df_test['workclass'] = df_test['workclass'].replace({
        'Federal-gov': 'Government', 
        'State-gov': 'Government', 
        'Local-gov': 'Government',
        'Self-emp-inc': 'Self-emp', 
        'Self-emp-not-inc': 'Self-emp'
    })
    df_val['workclass'] = df_val['workclass'].replace({
        'Federal-gov': 'Government', 
        'State-gov': 'Government', 
        'Local-gov': 'Government',
        'Self-emp-inc': 'Self-emp', 
        'Self-emp-not-inc': 'Self-emp'
    })
    df_holdout['workclass'] = df_holdout['workclass'].replace({
        'Federal-gov': 'Government', 
        'State-gov': 'Government', 
        'Local-gov': 'Government',
        'Self-emp-inc': 'Self-emp', 
        'Self-emp-not-inc': 'Self-emp'
    })

    # Creazione del dizionario per il mapping delle classi di lavoro
    unique_workclasses = pd.concat([df['workclass'], df_train['workclass'], df_test['workclass'], df_val['workclass'], df_holdout['workclass']]).unique()
    workclass_mapping = {label: idx for idx, label in enumerate(unique_workclasses)}

    # Applicazione del mapping
    df_train['workclass_enc'] = df_train['workclass'].map(workclass_mapping)
    df_test['workclass_enc'] = df_test['workclass'].map(workclass_mapping)
    df_val['workclass_enc'] = df_val['workclass'].map(workclass_mapping)
    df_holdout['workclass_enc'] = df_holdout['workclass'].map(workclass_mapping)

    # Eliminazione della colonna non encoded
    df_train.drop(columns=['workclass'], inplace=True)
    df_test.drop(columns=['workclass'], inplace=True)
    df_val.drop(columns=['workclass'], inplace=True)
    df_holdout.drop(columns=['workclass'], inplace=True)

    # Stampa le chiavi del mapping e i valori unici per debugging
    print("Chiavi del mapping:", workclass_mapping.keys())
    print("Valori unici in df_train['workclass']:", df_train['workclass'].unique())

        
        
    
    
    
    
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

    #encoding e dizionario 
    # Creazione del dizionario per il mapping delle classi di lavoro
    unique_education = df_train['education'].unique()
    
    education_mapping = {label: idx for idx, label in enumerate(unique_education)}
    inverse_education_mapping = {idx: label for label, idx in education_mapping.items()}

    #mapping
    df_train['education_enc'] = df_train['education'].apply(lambda x: education_mapping[x])
    df_test['education_enc'] = df_test['education'].apply(lambda x: education_mapping[x])
    df_val['education_enc'] = df_val['education'].apply(lambda x: education_mapping[x])
    df_holdout['education_enc'] = df_holdout['education'].apply(lambda x: education_mapping[x])
    # Eliminazione della feature non encoded
    df_train.drop(columns=['education'], inplace=True)
    df_test.drop(columns=['education'], inplace=True)
    df_val.drop(columns=['education'], inplace=True)
    df_holdout.drop(columns=['education'], inplace=True)
   
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

    #encoding e dizionario 
    # Creazione del dizionario per il mapping delle fasce istruzione
    education_num_mapping = {label: idx for idx, label in enumerate(labels_edu_num)}
    inverse_education_num_mapping = {idx: label for label, idx in education_num_mapping.items()}

    df_train['edu_num_enc'] = df_train['edu_num_group'].apply(lambda x: education_num_mapping[x])
    df_test['edu_num_enc'] = df_test['edu_num_group'].apply(lambda x: education_num_mapping[x])
    df_val['edu_num_enc'] = df_val['edu_num_group'].apply(lambda x: education_num_mapping[x])
    df_holdout['edu_num_enc'] = df_holdout['edu_num_group'].apply(lambda x: education_num_mapping[x])

    #eliminazione feature not encoded group, si puo ricavare l'encoded con la funzione
    df_train.drop(columns=['edu_num_group'], inplace=True)
    df_test.drop(columns=['edu_num_group'], inplace=True)
    df_val.drop(columns=['edu_num_group'], inplace=True)
    df_holdout.drop(columns=['edu_num_group'], inplace=True)
    
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

    
    # Create labels and mapping
    # Creazione delle etichette e del mapping
    labels_marital_status = df_train['marital-status'].unique().tolist()
    marital_status_mapping = {label: idx for idx, label in enumerate(labels_marital_status)}

    # Inversione del dizionario per accedere ai nomi dati i numeri
    inverse_marital_status_mapping = {idx: label for label, idx in marital_status_mapping.items()}

    # Encoding
    df_train['marital_status_enc'] = df_train['marital-status'].apply(lambda x: marital_status_mapping[x])
    df_test['marital_status_enc'] = df_test['marital-status'].apply(lambda x: marital_status_mapping[x])
    df_val['marital_status_enc'] = df_val['marital-status'].apply(lambda x: marital_status_mapping[x])
    df_holdout['marital_status_enc'] = df_holdout['marital-status'].apply(lambda x: marital_status_mapping[x])

    # Eliminazione della colonna originale
    df_train.drop(columns=['marital-status'], inplace=True)
    df_test.drop(columns=['marital-status'], inplace=True)
    df_val.drop(columns=['marital-status'], inplace=True)
    df_holdout.drop(columns=['marital-status'], inplace=True)
    
    
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


    # Create labels and mapping
    labels_occupation = df_train['occupation'].unique().tolist()
    occupation_mapping = {label: idx for idx, label in enumerate(labels_occupation)}

    # Inversione del dizionario per accedere ai nomi dati i numeri
    inverse_occupation_mapping = {idx: label for label, idx in occupation_mapping.items()}

    df_train['occupation_enc'] = df_train['occupation'].apply(lambda x: occupation_mapping[x])
    df_test['occupation_enc'] = df_test['occupation'].apply(lambda x: occupation_mapping[x])
    df_val['occupation_enc'] = df_val['occupation'].apply(lambda x: occupation_mapping[x])
    df_holdout['occupation_enc'] = df_holdout['occupation'].apply(lambda x: occupation_mapping[x])

    # Drop original 'occupation' column
    df_train.drop(columns=['occupation'], inplace=True)
    df_test.drop(columns=['occupation'], inplace=True)
    df_val.drop(columns=['occupation'], inplace=True)
    df_holdout.drop(columns=['occupation'], inplace=True)
    
 
    
    #RELATIONSHIP encode 
    # Create labels and mapping
    labels_relationship = df_train['relationship'].unique().tolist()
    relationship_mapping = {label: idx for idx, label in enumerate(labels_relationship)}

    # Inversione del dizionario per accedere ai nomi dati i numeri
    inverse_relationship_mapping = {idx: label for label, idx in relationship_mapping.items()}

    df_train['relationship_enc'] = df_train['relationship'].apply(lambda x: relationship_mapping[x])
    df_test['relationship_enc'] = df_test['relationship'].apply(lambda x: relationship_mapping[x])
    df_val['relationship_enc'] = df_val['relationship'].apply(lambda x: relationship_mapping[x])
    df_holdout['relationship_enc'] = df_holdout['relationship'].apply(lambda x: relationship_mapping[x])

    # Drop original column
    df_train.drop(columns=['relationship'], inplace=True)
    df_test.drop(columns=['relationship'], inplace=True)
    df_val.drop(columns=['relationship'], inplace=True)
    df_holdout.drop(columns=['relationship'], inplace=True)
    
    #RACE encode 
    # Create labels and mapping
    labels_race = df_train['race'].unique().tolist()
    race_mapping = {label: idx for idx, label in enumerate(labels_race)}

    # Inversione del dizionario per accedere ai nomi dati i numeri
    inverse_race_mapping = {idx: label for label, idx in race_mapping.items()}
    
    df_train['race_enc'] = df_train['race'].apply(lambda x: race_mapping[x])
    df_test['race_enc'] = df_test['race'].apply(lambda x: race_mapping[x])
    df_val['race_enc'] = df_val['race'].apply(lambda x: race_mapping[x])
    df_holdout['race_enc'] = df_holdout['race'].apply(lambda x: race_mapping[x])

    # Drop original column
    df_train.drop(columns=['race'], inplace=True)
    df_test.drop(columns=['race'], inplace=True)
    df_val.drop(columns=['race'], inplace=True)
    df_holdout.drop(columns=['race'], inplace=True)
    
    #SEX encode 
    # Create labels and mapping
    labels_sex = df_train['sex'].unique().tolist()
    sex_mapping = {label: idx for idx, label in enumerate(labels_sex)}

    # Inversione del dizionario per accedere ai nomi dati i numeri
    inverse_sex_mapping = {idx: label for label, idx in sex_mapping.items()}
    df_train['sex_enc'] = df_train['sex'].apply(lambda x: sex_mapping[x])
    df_test['sex_enc'] = df_test['sex'].apply(lambda x: sex_mapping[x])
    df_val['sex_enc'] = df_val['sex'].apply(lambda x: sex_mapping[x])
    df_holdout['sex_enc'] = df_holdout['sex'].apply(lambda x: sex_mapping[x])
    

    # Drop original column
    df_train.drop(columns=['sex'], inplace=True)
    df_test.drop(columns=['sex'], inplace=True)
    df_val.drop(columns=['sex'], inplace=True)
    df_holdout.drop(columns=['sex'], inplace=True)


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


    # Encoding delle categorie in valori numerici
    hours_per_week_mapping = {label: idx for idx, label in enumerate(labels_hours_per_week)}
    inverse_hours_per_week_mapping = {idx: label for label, idx in hours_per_week_mapping.items()}

    df_train['hours_per_week_enc'] = df_train['hours_per_week_group'].apply(lambda x: hours_per_week_mapping[x])
    df_test['hours_per_week_enc'] = df_test['hours_per_week_group'].apply(lambda x: hours_per_week_mapping[x])
    df_val['hours_per_week_enc'] = df_val['hours_per_week_group'].apply(lambda x: hours_per_week_mapping[x])
    df_holdout['hours_per_week_enc'] = df_holdout['hours_per_week_group'].apply(lambda x: hours_per_week_mapping[x])
    
    # Eliminare la colonna originale e la colonna categoriale 
    df_train.drop(columns=['hours-per-week', 'hours_per_week_group'], inplace=True)
    df_test.drop(columns=['hours-per-week', 'hours_per_week_group'], inplace=True)
    df_val.drop(columns=['hours-per-week', 'hours_per_week_group'], inplace=True)
    df_holdout.drop(columns=['hours-per-week', 'hours_per_week_group'], inplace=True)

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

    # Create labels and mapping
    labels_native_country = df_train['native-country'].unique().tolist()
    native_country_mapping = {label: idx for idx, label in enumerate(labels_native_country)}

    # Inversione del dizionario per accedere ai nomi dati i numeri
    inverse_native_country_mapping = {idx: label for label, idx in native_country_mapping.items()}
    #encoding
    df_train['native_country_enc'] = df_train['native-country'].apply(lambda x: native_country_mapping[x])
    df_test['native_country_enc'] = df_test['native-country'].apply(lambda x: native_country_mapping[x])
    df_val['native_country_enc'] = df_val['native-country'].apply(lambda x: native_country_mapping[x])
    df_holdout['native_country_enc'] = df_holdout['native-country'].apply(lambda x: native_country_mapping[x])

    # Drop original column
    df_train.drop(columns=['native-country'], inplace=True)
    df_test.drop(columns=['native-country'], inplace=True)
    df_val.drop(columns=['native-country'], inplace=True)
    df_holdout.drop(columns=['native-country'], inplace=True)

    
    return df_train, df_test, df_val, df_holdout

