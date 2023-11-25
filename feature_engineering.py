import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score

train = pd.read_csv('train.csv', sep = ',')
validate = pd.read_csv('validate.csv', sep = ',')
test = pd.read_csv('test.csv', sep = ',')

################################################# Create new features #################################################

train['attempted_scan'] = np.where(train['scansWithoutRegistration'] > 0, 1, 0)
validate['attempted_scan'] = np.where(validate['scansWithoutRegistration'] > 0, 1, 0)
test['attempted_scan'] = np.where(test['scansWithoutRegistration'] > 0, 1, 0)

train['made_modification'] = np.where(train['quantityModifications'] > 0, 1, 0)
validate['made_modification'] = np.where(validate['quantityModifications'] > 0, 1, 0)
test['made_modification'] = np.where(test['quantityModifications'] > 0, 1, 0)

train['total_items_scanned'] = train['totalScanTimeInSeconds'] * train['scannedLineItemsPerSecond']
validate['total_items_scanned'] = validate['totalScanTimeInSeconds'] * validate['scannedLineItemsPerSecond']
test['total_items_scanned'] = test['totalScanTimeInSeconds'] * test['scannedLineItemsPerSecond']

train['value_per_item'] = train['grandTotal'] / train['total_items_scanned']
validate['value_per_item'] = validate['grandTotal'] / validate['total_items_scanned']
test['value_per_item'] = test['grandTotal'] / test['total_items_scanned']

train['ratio_of_voided_scans'] = train['lineItemVoids'] / train['total_items_scanned']
validate['ratio_of_voided_scans'] = validate['lineItemVoids'] / validate['total_items_scanned']
test['ratio_of_voided_scans'] = test['lineItemVoids'] / test['total_items_scanned']

################################################# before feature preprocessing #################################################
feature_cols = [col for col in train.columns if col != 'fraud']
target_col = ['fraud']
X_train = train[feature_cols].copy()
y_train = train[target_col].copy()
X_validate = validate[feature_cols].copy()
y_validate = validate[target_col].copy()
X_test  = test[feature_cols].copy()
y_test  = test[target_col].copy()

LogR = LogisticRegression(class_weight='balanced') # for imbalanced learning
LogR.fit(X_train, y_train)

print('Before feature preprocessing:')
print('F1 score over training dataset: {:0.4f}'.format(f1_score(y_train, LogR.predict(X_train),  average='macro')))
print('F1 score over validation dataset: {:0.4f}'.format(f1_score(y_validate, LogR.predict(X_validate),  average='macro')))
print('F1 score over test dataset: {:0.4f}'.format(f1_score(y_test, LogR.predict(X_test),  average='macro')))

################################################# bin trustLevel #################################################

def bin_trust(trust_level):
    if trust_level < 3:
        return 'likely_fraud'
    else:
        return 'unlikely_fraud'

train['trustLevel'] = train['trustLevel'].apply(bin_trust)
validate['trustLevel'] = validate['trustLevel'].apply(bin_trust)
test['trustLevel'] = test['trustLevel'].apply(bin_trust)

############################################# encode likely and unlikely fraud #############################################

encoder = OneHotEncoder(handle_unknown="ignore")

# Fit encoder on training data (returns a separate DataFrame)
data_ohe = pd.DataFrame(encoder.fit_transform(train[["trustLevel"]]).toarray())
data_ohe.columns = [col for cols in encoder.categories_ for col in cols]

# Join encoded data with original training data
train = pd.concat([train, data_ohe], axis=1)

# Transform validation data
data_ohe = pd.DataFrame(encoder.transform(validate[["trustLevel"]]).toarray())
data_ohe.columns = [col for cols in encoder.categories_ for col in cols]
validate = pd.concat([validate, data_ohe], axis=1)

# Transform test data
data_ohe = pd.DataFrame(encoder.transform(test[["trustLevel"]]).toarray())
data_ohe.columns = [col for cols in encoder.categories_ for col in cols]
test = pd.concat([test, data_ohe], axis=1)

train = train.drop(columns='trustLevel')
validate = validate.drop(columns='trustLevel')
test = test.drop(columns='trustLevel')

################################################# scale continuour variables #################################################

all_cols = [col for col in train.columns if (col != 'likely_fraud') and (col != 'unlikely_fraud') and 
            (col != 'fraud') and (col != 'attempted_scan') and (col != 'made_modification')]
new_col_names = [f'scaled_{col}' for col in all_cols]

scaler = StandardScaler().fit(train[all_cols])
train[new_col_names] = scaler.transform(train[all_cols])
validate[new_col_names] = scaler.transform(validate[all_cols])
test[new_col_names] = scaler.transform(test[all_cols])

train = train.drop(columns = all_cols)
validate = validate.drop(columns = all_cols)
test = test.drop(columns = all_cols)

################################################# after feature preprocessing #################################################

feature_cols = [col for col in train.columns if col != 'fraud']
target_col = ['fraud']
X_train = train[feature_cols].copy()
y_train = train[target_col].copy()
X_validate = validate[feature_cols].copy()
y_validate = validate[target_col].copy()
X_test  = test[feature_cols].copy()
y_test  = test[target_col].copy()

LogR = LogisticRegression(class_weight='balanced') # for imbalanced learning
LogR.fit(X_train, y_train)

print('After feature preprocessing:')
print('F1 score over training dataset: {:0.4f}'.format(f1_score(y_train, LogR.predict(X_train),  average='macro')))
print('F1 score over validation dataset: {:0.4f}'.format(f1_score(y_validate, LogR.predict(X_validate),  average='macro')))
print('F1 score over test dataset: {:0.4f}'.format(f1_score(y_test, LogR.predict(X_test),  average='macro')))
