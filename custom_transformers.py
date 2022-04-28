from sklearn.base import BaseEstimator, TransformerMixin


class CyclesToFailureAdder(BaseEstimator, TransformerMixin):
    def __init__(self): ...

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['cycles_to_failure'] = X.groupby('id')['cycle'].transform('last')
        X['cycles_to_failure'] = X['cycles_to_failure'] - X['cycle']
        X = X.drop(columns=['p00', 'p01', 'p07', 'p09', 'p10', 'p16', 's1', 's2'])  # also drop unused attributes
        return X
