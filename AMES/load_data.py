import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

SEED = 202042
"""
A script to load in ISSSTY data from csv file. 
Output is toxicity values (either MTL or overall) for all molecules
plus molecule ID number



"""

def load_data(data_path, model):
    # load data
    df = pd.read_csv(data_path)

    # PARTITIONS
    train = df.loc[df['Partition'].str.contains('Train')]
    internal = df.loc[df['Partition'].str.contains('Internal')]
    external = df.loc[df['Partition'].str.contains('External')]

    # Target values per partition - MTL
    y_train_MTL = train[['TA98', 'TA100', 'TA102', 'TA1535', 'TA1537']]
    y_internal_MTL = internal[['TA98', 'TA100', 'TA102', 'TA1535', 'TA1537']]
    y_external_MTL = external[['TA98', 'TA100', 'TA102', 'TA1535', 'TA1537']]

    mol_ID_train = train['Id']
    mol_ID_internal = internal['Id']
    mol_ID_external = external['Id']

    # Target values per partition - Overall
    y_train_overall = train['Overall']
    y_internal_overall = internal['Overall']
    y_external_overall = external['Overall']

    output = ()

    y_train_MTL = [y_train_MTL.iloc[:, 0].values,
                    y_train_MTL.iloc[:, 1].values,
                    y_train_MTL.iloc[:, 2].values,
                    y_train_MTL.iloc[:, 3].values,
                    y_train_MTL.iloc[:, 4].values]

    y_internal_MTL = [y_internal_MTL.iloc[:, 0].values,
                        y_internal_MTL.iloc[:, 1].values,
                        y_internal_MTL.iloc[:, 2].values,
                        y_internal_MTL.iloc[:, 3].values,
                        y_internal_MTL.iloc[:, 4].values]

    y_external_MTL = [y_external_MTL.iloc[:, 0].values,
                      y_external_MTL.iloc[:, 1].values,
                      y_external_MTL.iloc[:, 2].values,
                      y_external_MTL.iloc[:, 3].values,
                      y_external_MTL.iloc[:, 4].values]


    labels_dict = {
        'MTL': (y_train_MTL, y_internal_MTL, y_external_MTL),
        'Overall': (y_train_overall.values, y_internal_overall.values, y_external_overall.values)
    }

    y = labels_dict.get(model)

    t = (mol_ID_train, y[0])
    v = (mol_ID_internal, y[1])
    te = (mol_ID_external, y[2])
    # In this case, the output is a tuple containing the train and internal partitions
    output = (t, v, te)

    return output