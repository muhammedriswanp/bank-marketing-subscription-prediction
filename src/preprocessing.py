from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

# Columns to drop entirely
DROP_COLS = ['duration',     # Data Leakage
                'default',      # near-zero variance (only 3 yes values)
                'pdays'         # replaced by previous_contact feature
                ]

# Numerical columns to scale
NUMERICAL_COLS = [
    'age',
    'campaign',
    'previous',
    'emp.var.rate',       
    'cons.price.idx',
    'cons.conf.idx',
    'euribor3m',
    'nr.employed',
    'previous_contact'
]

# Categorical columns to encode
CATEGORICAL_COLS = [
    'job',
    'marital',
    'education',
    'housing',
    'loan',
    'contact',
    'month',
    'day_of_week',
    'poutcome'
]

numerical_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]
)

