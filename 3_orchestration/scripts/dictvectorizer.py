from sklearn.feature_extraction import DictVectorizer
from typing import Dict, List, Optional, Tuple
import pandas as pd
import scipy



if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


# def transformation():
#     categorical = ['PULocationID', 'DOLocationID']
#     # numerical = ['trip_distance']

#     dv = DictVectorizer()

#     train_dicts = df_train[categorical].to_dict(orient='records')
#     X_train = dv.fit_transform(train_dicts)

#     val_dicts = df_val[categorical].to_dict(orient='records')
#     X_val = dv.transform(val_dicts)

@transformer
def vectorize_features(training_set: pd.DataFrame, validation_set: d.DataFrame) -> Tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix, DictVectorizer]:

    dv = DictVectorizer()

    train_dicts = training_set.to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)

    # val_dicts = validation_set.to_dict(orient='records')
    # X_val = dv.transform(val_dicts)

    val_dicts = validation_set[training_set.columns].to_dict(orient='records')
    X_val = dv.transform(val_dicts)


    return X_train, X_val, dv


# @test
# def test_output(output, *args) -> None:
#     """
#     Template code for testing the output of the block.
#     """
#     assert output is not None, 'The output is undefined'