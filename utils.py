################################################
# IMPORTS
################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import re
from IPython.display import display
import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRegressor # type: ignore
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, cross_val_score, GridSearchCV, StratifiedKFold, train_test_split, validation_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import FunctionTransformer, StandardScaler, RobustScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, f1_score, confusion_matrix, classification_report
from xgboost import XGBClassifier, XGBRegressor # type: ignore
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score
from imblearn.over_sampling import SMOTE


################################################
# 1. DATA LOADING & BASIC HELPERS
################################################


def load(dataset, sep=None, decimal=None):
    """
    Loads a CSV file.
    If sep or decimal are not provided, the file is read automatically.
    """
    try:
        if sep and decimal:
            data = pd.read_csv(dataset, sep=sep, decimal=decimal)
        else:
            data = pd.read_csv(dataset)  # default: sep=",", decimal="."
        return data
    except FileNotFoundError:
        print(f"File not found: {dataset}")
        return None
    except Exception as e:
        print(f"Error occurred ({dataset}): {e}")
        return None


################################################
# 2. MISSING VALUE HANDLING
################################################

def table_missing_values(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


def missing_VS_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")
        

def missing_target_heatmap(df, target, na_columns):
    temp_df = df.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = temp_df[col].isnull().astype(int)
    
    na_flags = [c for c in temp_df.columns if "_NA_FLAG" in c]
    mean_target = [temp_df.groupby(col)[target].mean()[1] if 1 in temp_df.groupby(col)[target].mean().index else 0 
                   for col in na_flags]

    plt.figure(figsize=(10, len(na_flags)*0.5))
    sns.heatmap(np.array(mean_target).reshape(-1,1), annot=True, cmap="YlOrRd", cbar=True,
                yticklabels=[c.replace("_NA_FLAG","") for c in na_flags])
    plt.title(f"Effect of NA vs {target}")
    plt.xlabel("TARGET_MEAN")
    plt.show()

# Add NA_FLAG columns for missing values and fill with median or mode
def handle_missing_values(df):
    df_clean = df.copy()
    # Numeric columns (flag + median)
    num_cols_flag = ['shipping_time', 'prep_time', 'distance_km', 'order_approved_at', 'estimated_time']
    for col in num_cols_flag:
        na_flag_col = col + "_NA_FLAG"
        df_clean[na_flag_col] = df_clean[col].isnull().astype(int)
        median_val = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median_val)
    # Columns with very few missing values -> drop
    cols_few_na = ['payment_types', 'payment_value_sum', 'payment_installments_max']
    df_clean = df_clean.dropna(subset=cols_few_na)
    return df_clean


def handle_missing_values_leakfree(df) -> pd.DataFrame:
    df = df.copy()
    for col in ["prep_time", "estimated_time", "distance_km"]:
        if col in df.columns:
            df[col + "_NA_FLAG"] = df[col].isnull().astype(int)
            df[col] = df[col].fillna(df[col].median())
    for col in ["payment_types", "payment_value_sum", "payment_installments_max"]:
        if col in df.columns:
            df = df.dropna(subset=[col])
    return df

################################################
# 3. FEATURE CATEGORIZATION
################################################
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Returns the names of categorical, numerical, and categorical but cardinal variables in the dataset.
    Note: Numerical-looking categorical variables are also included in categorical variables.

    Parameters
    ------
        dataframe: dataframe
            The dataframe to extract column names from
        cat_th: int, optional
            Threshold for numerical variables that are actually categorical
        car_th: int, optional
            Threshold for categorical but cardinal variables

    Returns
    ------
        cat_cols: list
            List of categorical variables
        num_cols: list
            List of numerical variables
        cat_but_car: list
            List of categorical-looking cardinal variables

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))

    Notes
    ------
        cat_cols + num_cols + cat_but_car = total number of variables
        num_but_cat is included in cat_cols.
        The sum of returned three lists equals the total number of variables: 
        cat_cols + num_cols + cat_but_car = number of columns
    """

    # Identify categorical and categorical-but-cardinal columns
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # Identify numerical columns
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    return cat_cols, num_cols, cat_but_car



################################################
# 4. OUTLIER DETECTION & HANDLING
################################################

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    """
    Returns lower and upper thresholds for outliers in a column based on IQR
    """
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    """
    Replaces outliers in a column with threshold values
    """
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    """
    Checks if a single column has outliers
    Returns True/False
    """
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    # Only check numeric columns
    if dataframe[col_name].dtype.kind in 'bifc':  # boolean, integer, float, complex
        if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
            return True
        else:
            return False
    else:
        return False 


def check_outliers_for_columns(dataframe, cols_list, q1=0.25, q3=0.75):
    """
    Returns a dictionary indicating if each column has outliers
    """
    result = {}
    for col in cols_list:
        result[col] = check_outlier(dataframe, col, q1, q3)
    return result


def check_outlier_cleaning(original_df, clean_df, cols):
    """
    Visual comparison of original and cleaned columns for outliers
    """
    for col in cols:
        print(f"\n=== {col} ===")
        
        # Display describe() results side by side
        compare_df = pd.DataFrame({
            "Original": original_df[col].describe(),
            "Cleaned": clean_df[col].describe()
        })
        display(compare_df)
        
        print(f"Negative values after cleaning: {(clean_df[col] < 0).sum()}")
        print(f"Missing values: {clean_df[col].isnull().sum()}")
        
        # Side-by-side visualization: boxplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.boxplot(x=original_df[col], ax=axes[0], color="salmon")
        axes[0].set_title(f"Original {col}")
        sns.boxplot(x=clean_df[col], ax=axes[1], color="seagreen")
        axes[1].set_title(f"Cleaned {col}")
        plt.show()
        
        # Side-by-side histograms
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(original_df[col], bins=50, kde=True, ax=axes[0], color="salmon")
        axes[0].set_title(f"Original {col}")
        sns.histplot(clean_df[col], bins=50, kde=True, ax=axes[1], color="seagreen")
        axes[1].set_title(f"Cleaned {col}")
        plt.show()


################################################
# 5. ENCODING
################################################

def encode_label(dataframe, binary_col):
    """
    Label encoding for binary columns
    """
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    """
    One-hot encoding for multiple categorical columns
    """
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

################################################
# 6. FEATURE ENGINEERING
################################################

def add_time_features_leakfree(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds time-based features without introducing leakage.
    """
    df = df.copy()
    if {"order_purchase_timestamp", "order_approved_at"} <= set(df.columns):
        df["prep_time"] = (df["order_approved_at"] - df["order_purchase_timestamp"]).dt.total_seconds() / 3600
    if {"order_estimated_delivery_date", "order_purchase_timestamp"} <= set(df.columns):
        df["estimated_time"] = (df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]).dt.total_seconds() / 3600
    if "order_purchase_timestamp" in df.columns:
        ts = df["order_purchase_timestamp"]
        df["year"] = ts.dt.year
        df["month"] = ts.dt.month
        df["week"] = ts.dt.isocalendar().week.astype(int)
        df["dayofweek"] = ts.dt.dayofweek
        df["hour"] = ts.dt.hour
    return df


def add_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds derived features such as normalized distance, weekend flag, time bins, and combined day-hour feature.
    """
    df = df.copy()
    if "distance_km" in df.columns:
        df["distance_norm"] = np.log1p(df["distance_km"])
    if "dayofweek" in df.columns:
        df["is_weekend"] = df["dayofweek"].isin([5,6]).astype(int)
    if "hour" in df.columns:
        df["time_bin"] = pd.cut(df["hour"], bins=[0,6,12,18,24], labels=['night','morning','afternoon','evening'])
    if {"dayofweek","hour"} <= set(df.columns):
        df["dow_hour"] = df["dayofweek"]*24 + df["hour"]
    return df


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops columns that could lead to data leakage.
    """
    drop_cols = [
        "order_delivered_carrier_date", "order_delivered_customer_date",
        "estimated_time","shipping_time", 'prep_time','shipping_time_NA_FLAG', 'estimated_time_NA_FLAG', 
        'prep_time_NA_FLAG',"delay_days", "delay_class", "is_late",
        "order_purchase_timestamp", "order_approved_at", "order_estimated_delivery_date"
    ]
    return df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")


################################################
# 7. ID ENCODING
################################################

def _safe_str_id(s): 
    return s.astype("string").fillna("_nan_")
    
def fit_id_stats(X_train, y_train, id_col, m=50):
    """
    Computes ID-based statistics: frequency and late_rate (smoothed).
    """
    y_train = y_train.loc[X_train.index]
    ids = _safe_str_id(X_train[id_col])
    g = pd.DataFrame({"id": ids, "y": y_train}).groupby("id")["y"].agg(["sum","count"])
    global_mean = y_train.mean()
    g["late_rate"] = (g["sum"] + m*global_mean) / (g["count"] + m)
    return {"freq": g["count"], "late_rate": g["late_rate"], "global_mean": global_mean}
    
def apply_id_stats(X, id_col, stats, prefix):
    """
    Applies previously computed ID statistics to a dataset.
    """
    X = X.copy()
    if id_col not in X.columns:
        return X
    ids = _safe_str_id(X[id_col])
    X[f"{prefix}_freq"] = ids.map(stats["freq"]).fillna(0).astype("int64")
    X[f"{prefix}_late_rate"] = ids.map(stats["late_rate"]).fillna(stats["global_mean"]).astype("float64")
    return X.drop(columns=[id_col])


id_plan = [("cusUni","cust"), ("seller_id_pref","seller"), ("product_id_pref","product")]
def add_all_id_encodings(X_train, X_test, y_train, id_plan=id_plan, m=50):
    """
    Encodes multiple ID columns with frequency and smoothed late_rate.
    """
    Xtr, Xte = X_train.copy(), X_test.copy()
    id_plan = [(c,p) for c,p in id_plan if c in Xtr.columns]
    learned = {}
    for id_col, prefix in id_plan:
        stats = fit_id_stats(Xtr, y_train, id_col, m=m)
        learned[id_col] = stats
        Xtr = apply_id_stats(Xtr, id_col, stats, prefix)
        Xte = apply_id_stats(Xte, id_col, stats, prefix)
    drop_candidates = ["cusUni","seller_id_pref","product_id_pref","customer_order_id","order_id_pref"]
    Xtr = Xtr.drop(columns=[c for c in drop_candidates if c in Xtr.columns], errors="ignore")
    Xte = Xte.drop(columns=[c for c in drop_candidates if c in Xte.columns], errors="ignore")
    return Xtr, Xte, learned


################################################
# 8. CUSTOM TRANSFORMERS
################################################

class PaymentTypeEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes payment_types column into binary features for each type and dominant payment type.
    """
    def __init__(self, col="payment_types", sep="/", drop_original=True, add_dominant=True,
                 risk_order=("boleto","debit_card","voucher","credit_card"), normalize_map=None):
        self.col = col
        self.sep = sep
        self.drop_original = drop_original
        self.add_dominant = add_dominant
        self.risk_order = risk_order
        self.normalize_map = normalize_map or {"credit card":"credit_card","debit card":"debit_card",
                                               "boleta":"boleto","baleto":"boleto","voucher ":"voucher"}
        self.classes_ = []

    def _split_clean(self,x):
        if pd.isna(x) or str(x).strip()=="": return []
        toks = [self.normalize_map.get(t.strip().lower(), t.strip().lower()) for t in str(x).split(self.sep)]
        toks = [re.sub(r"[^a-z0-9_]+","",t) for t in toks if t]
        return sorted(set(toks))

    def _dominant(self,lst):
        s = set(lst)
        for t in self.risk_order:
            if t in s: return t
        return None

    def fit(self,X,y=None):
        if self.col not in X.columns: 
            self.classes_ = []
            return self
        lists = X[self.col].apply(self._split_clean)
        all_types = set()
        for lst in lists: all_types.update(lst)
        self.classes_ = sorted(all_types)
        return self

    def transform(self,X):
        X = X.copy()
        if self.col in X.columns:
            lists = X[self.col].apply(self._split_clean)
        else:
            lists = pd.Series([[]]*len(X), index=X.index)
        for t in self.classes_: X[t] = lists.apply(lambda lst: int(t in lst))
        if self.add_dominant: X["dominant_payment_type"] = lists.apply(self._dominant)
        if self.drop_original and self.col in X.columns: X = X.drop(columns=[self.col])
        return X


################################################
# 9. PREPROCESSING PIPELINES
################################################

NUM_LOG_ROBUST_CANDIDATES = ["prep_time","estimated_time","distance_km","payment_value_sum"]
NUM_STD_CANDIDATES = ["payment_installments_max"]
OHE_CANDIDATES = ["year","month","week","seller_count_per_order"]

clip_nonneg = FunctionTransformer(lambda z: np.clip(z, a_min=0, a_max=None), validate=False)
log1p = FunctionTransformer(np.log1p, validate=False)

num_log_robust = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("clip0", clip_nonneg),
    ("log1p", log1p),
    ("scale", RobustScaler())
])
num_std = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scale", StandardScaler())
])
cat_ohe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])


def build_full_preprocessor_leakfree(X_sample):
    """
    Constructs a preprocessing pipeline with numeric, categorical, and payment-type features.
    """
    num_lr_cols  = [c for c in NUM_LOG_ROBUST_CANDIDATES if c in X_sample.columns]
    num_std_cols = [c for c in NUM_STD_CANDIDATES if c in X_sample.columns]
    ohe_cols = [c for c in OHE_CANDIDATES if c in X_sample.columns] + ["dominant_payment_type"]

    pre = ColumnTransformer([
        ("num_lr", num_log_robust, num_lr_cols),
        ("num_std", num_std, num_std_cols),
        ("ohe", cat_ohe, ohe_cols)
    ], remainder="passthrough")

    pipe = Pipeline([
        ("payfeat", PaymentTypeEncoder()),
        ("prep", pre)
    ])
    return pipe

################################################
# 10. LEAKAGE CHECKS & DISTRIBUTIONS
################################################

def plot_late_rate_distributions(X_train, X_test, cols=["cust_late_rate", "product_late_rate"]):
    """
    Compares distributions of late_rate features between train and test sets.
    """
    ncols = len(cols)
    plt.figure(figsize=(6 * ncols, 5))
    
    for i, col in enumerate(cols, 1):
        plt.subplot(1, ncols, i)
        sns.kdeplot(X_train[col], label="Train", fill=True, alpha=0.5)
        sns.kdeplot(X_test[col], label="Test", fill=True, alpha=0.5)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.legend()
    
    plt.tight_layout()
    plt.show()


def compare_late_rate_features(X_train, X_test):
    """
    Compares train and test distributions of all late_rate columns and shows KS and Wasserstein distances.
    """
    late_rate_cols = [c for c in X_train.columns if "late_rate" in c]
    if not late_rate_cols:
        print("No late_rate columns found.")
        return
    
    n_cols = 2
    n_rows = int(np.ceil(len(late_rate_cols) / n_cols))
    plt.figure(figsize=(6*n_cols, 4*n_rows))
    
    for i, col in enumerate(late_rate_cols, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.kdeplot(X_train[col], label="Train", fill=True, alpha=0.5)
        sns.kdeplot(X_test[col], label="Test", fill=True, alpha=0.5)
        ks = ks_2samp(X_train[col].dropna(), X_test[col].dropna()).statistic
        w = wasserstein_distance(X_train[col].dropna(), X_test[col].dropna())
        plt.title(f"{col}\nKS={ks:.3f}, W={w:.3f}")
        plt.legend()
    
    plt.tight_layout()
    plt.show()


def compare_feature_distributions(X_train, X_test, max_plots=20):
    """
    Compares numeric feature distributions between train and test sets.
    Returns a DataFrame with KS and Wasserstein distances.
    """
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    results = []

    for col in numeric_cols:
        train_vals = X_train[col].dropna()
        test_vals = X_test[col].dropna()
        ks = ks_2samp(train_vals, test_vals).statistic
        w = wasserstein_distance(train_vals, test_vals)
        results.append({"feature": col, "KS": ks, "Wasserstein": w})
    
    results_df = pd.DataFrame(results).sort_values("KS", ascending=False)

    n_cols = 2
    n_rows = int(np.ceil(min(max_plots, len(results_df)) / n_cols))
    plt.figure(figsize=(6*n_cols, 4*n_rows))

    for i, col in enumerate(results_df["feature"].head(max_plots), 1):
        plt.subplot(n_rows, n_cols, i)
        sns.kdeplot(X_train[col], label="Train", fill=True, alpha=0.5)
        sns.kdeplot(X_test[col], label="Test", fill=True, alpha=0.5)
        plt.title(f"{col}\nKS={results_df.loc[results_df.feature==col, 'KS'].values[0]:.3f}, "
                  f"W={results_df.loc[results_df.feature==col, 'Wasserstein'].values[0]:.3f}")
        plt.legend()

    plt.tight_layout()
    plt.show()

    return results_df


def leakage_check(X: pd.DataFrame, y: pd.Series, top_n=20):
    """
    Calculates correlation between features and target (numeric + categorical).
    Returns top_n features with highest correlation.
    """
    df = X.copy()
    df["__target__"] = y.values
    corr_results = {}

    for col in df.drop(columns="__target__").columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            corr = np.corrcoef(df[col], df["__target__"])[0,1]
        else:
            means = df.groupby(col)["__target__"].mean()
            encoded = df[col].map(means)
            corr = np.corrcoef(encoded, df["__target__"])[0,1]
        corr_results[col] = abs(corr)

    corr_df = pd.DataFrame.from_dict(corr_results, orient="index", columns=["abs_corr"])
    corr_df = corr_df.sort_values("abs_corr", ascending=False).head(top_n)
    return corr_df


################################################
# 11. BASE MODELS & PIPELINE TESTS
################################################

def base_models(X, y, scoring="roc_auc"):
    """
    Evaluates basic classifiers with cross-validation.
    """
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('XGBoost', XGBClassifier(eval_metric='logloss')),           
                   ('LightGBM', LGBMClassifier(verbosity=-1, force_row_wise=True))]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")


def base_models_pipeline(X, y, preprocessor, scoring="roc_auc", cv=3):
    """
    Evaluates base classifiers within a preprocessing pipeline.
    """
    print("Base Models...\n")
    classifiers = [
        ("LR", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ("CART", DecisionTreeClassifier()),
        ("RF", RandomForestClassifier(n_estimators=200)),
        ("LightGBM", LGBMClassifier()),
        ("XGBoost", XGBClassifier(eval_metric='logloss', verbose=-1))
    ]
    
    results = {}
    for name, clf in classifiers:
        pipe = Pipeline([
            ("prep", preprocessor),
            ("model", clf)
        ])
        cv_results = cross_validate(pipe, X, y, cv=cv, scoring=scoring)
        mean_score = cv_results['test_score'].mean()
        print(f"{name} {scoring}: {mean_score:.4f}")
        results[name] = mean_score
    
    return results

################################################
# 12. HYPERPARAMETER OPTIMIZATION
################################################

# Model parameters
knn_params = {"n_neighbors": range(2, 50)}

cart_params = { 
    'max_depth': [5, 10, 15],
    "min_samples_split": [2, 5, 10]
}

rf_params = {
    "max_depth": [8, 15, None],
    "max_features": [5, 7],
    "min_samples_split": [15, 20],
    "n_estimators": [200, 300]
}

xgboost_params = {
    "learning_rate": [0.01, 0.1],
    "max_depth": [5, 8],
    "n_estimators": [100, 200],
    "colsample_bytree": [0.5, 1]
}

lightgbm_params = {
    "learning_rate": [0.01, 0.1],
    "n_estimators": [300, 500],
    "colsample_bytree": [0.7, 1]
}

# Updated classifier list
classifiers = [
    # ('KNN', KNeighborsClassifier(), knn_params),  # optional
    ("CART", DecisionTreeClassifier(), cart_params),
    ("RF", RandomForestClassifier(), rf_params),
    ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
    ('LightGBM', LGBMClassifier(verbosity=-1, force_row_wise=True), lightgbm_params)
]


def hyperparameter_optimization_processed(X_train, y_train, X_test, y_test, classifiers, cv=3, scoring="roc_auc"):
    print("Hyperparameter Optimization on Processed Data (without Pipeline)...\n")
    best_models = {}
    model_scores = []

    for name, model, params in classifiers:
        print(f"########## {name} ##########")

        # Initial CV score
        cv_score = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
        print(f"{scoring} (CV Before): {cv_score.mean():.4f}")

        # GridSearchCV
        gs = GridSearchCV(model, params, cv=cv, n_jobs=-1, verbose=1, scoring=scoring)
        gs.fit(X_train, y_train)

        # Best model
        best_model = gs.best_estimator_

        # CV score after tuning
        cv_after = cross_val_score(best_model, X_train, y_train, cv=cv, scoring=scoring)
        print(f"{scoring} (CV After): {cv_after.mean():.4f}")

        # Test score
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        test_score = roc_auc_score(y_test, y_pred_proba)
        print(f"{scoring} (Test): {round(test_score, 4)}")

        # Best parameters
        print(f"{name} best params: {gs.best_params_}\n")

        # Save results to dictionary
        model_scores.append({
            "Model": name,
            "CV_Before": cv_score.mean(),
            "CV_After": cv_after.mean(),
            "Test": test_score
        })

        best_models[name] = best_model

    # Convert results to DataFrame
    scores_df = pd.DataFrame(model_scores)
    return best_models, scores_df

def hyperparameter_optimization_processed_with_threshold(X_train, y_train, X_test, y_test, classifiers, cv=3, scoring="roc_auc"):
    
    print("Hyperparameter Optimization on Processed Data with Optimal Threshold...\n")
    best_models = {}
    model_scores = []

    for name, model, params in classifiers:
        print(f"########## {name} ##########")

        # Initial CV score (default parameters)
        cv_score = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
        print(f"{scoring} (CV Before): {round(cv_score['test_score'].mean(), 4)}")

        # GridSearchCV
        gs = GridSearchCV(model, params, cv=cv, n_jobs=-1, verbose=1, scoring=scoring)
        gs.fit(X_train, y_train)

        # Best model
        best_model = gs.best_estimator_

        # CV score after tuning
        cv_after = cross_validate(best_model, X_train, y_train, cv=cv, scoring=scoring)
        print(f"{scoring} (CV After): {round(cv_after['test_score'].mean(), 4)}")

        # Predicted probabilities on test set
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]

        # Test ROC-AUC score
        test_roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"{scoring} (Test): {round(test_roc_auc, 4)}")

        # Optimal F1 threshold
        thresholds = np.linspace(0, 1, 100)
        f1_scores = [f1_score(y_test, y_pred_proba >= t) for t in thresholds]
        best_thresh = thresholds[np.argmax(f1_scores)]
        print(f"Optimal F1 threshold: {best_thresh:.3f}")
        print(f"F1 at optimal threshold: {f1_score(y_test, y_pred_proba >= best_thresh):.4f}\n")

        # Save model and scores
        model_scores.append({
            "Model": name,
            "CV_Before": cv_score["test_score"].mean(),
            "CV_After": cv_after["test_score"].mean(),
            "Test": test_roc_auc
        })

        best_models[name] = {
            "model": best_model,
            "optimal_threshold": best_thresh,
            "roc_auc": test_roc_auc,
            "f1_at_threshold": f1_score(y_test, y_pred_proba >= best_thresh)
        }

    scores_df = pd.DataFrame(model_scores)
    return best_models, scores_df
# Multi-class classifier parameters
cart_params_multi = {
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

# Random Forest parameters for multi-class
rf_params_multi = {
    "n_estimators": [100, 200, 300],
    "max_depth": [8, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["auto", "sqrt"]
}

# XGBoost parameters for multi-class
xgboost_params_multi = {
    "n_estimators": [100, 200],
    "learning_rate": [0.01, 0.1],
    "max_depth": [5, 8],
    "colsample_bytree": [0.5, 1]
}

# LightGBM parameters for multi-class
lightgbm_params_multi = {
    "n_estimators": [300, 500],
    "learning_rate": [0.01, 0.1],
    "num_leaves": [31, 50, 100],
    "colsample_bytree": [0.7, 1]
}

# Multi-class classifier list
multi_classifiers = [
    # ('KNN', KNeighborsClassifier(), knn_params),
    ("CART", DecisionTreeClassifier(), cart_params_multi),
    ("RF", RandomForestClassifier(), rf_params_multi),
    ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params_multi),
    ('LightGBM', LGBMClassifier(verbosity=-1, force_row_wise=True), lightgbm_params_multi)
]


def hyperparameter_optimization_multiclass(X_train, y_train, X_test, y_test, multi_classifiers, cv=3, scoring="accuracy"):         
    print("Hyperparameter Optimization for Multi-class Classification...\n")
    best_models = {}
    model_scores = []

    for name, model, params in multi_classifiers:
        print(f"########## {name} ##########")

        # Initial CV score
        cv_score = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
        print(f"{scoring} (CV Before): {cv_score.mean():.4f}")

        # GridSearchCV
        gs = GridSearchCV(model, params, cv=cv, n_jobs=-1, verbose=1, scoring=scoring)
        gs.fit(X_train, y_train)

        best_model = gs.best_estimator_

        # CV score after tuning
        cv_after = cross_val_score(best_model, X_train, y_train, cv=cv, scoring=scoring)
        print(f"{scoring} (CV After): {cv_after.mean():.4f}")

        # ---- Test set score ----
        y_pred = best_model.predict(X_test)
        test_score = accuracy_score(y_test, y_pred)
        print(f"{scoring} (Test): {test_score:.4f}\n")

        # Save results to dictionary
        model_scores.append({
            "Model": name,
            "CV_Before": cv_score.mean(),
            "CV_After": cv_after.mean(),
            "Test": test_score
        })

        best_models[name] = best_model

    # Convert results to DataFrame
    scores_df = pd.DataFrame(model_scores)
    return best_models, scores_df

##--- REGRESSION

# CART (Decision Tree Regressor) parameters
cart_params_reg = {
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

# Random Forest Regressor parameters
rf_params_reg = {
    "n_estimators": [100, 200, 300],
    "max_depth": [8, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["auto", "sqrt"]
}

# XGBoost Regressor parameters
xgboost_params_reg = {
    "n_estimators": [100, 200],
    "learning_rate": [0.01, 0.1],
    "max_depth": [5, 8],
    "colsample_bytree": [0.5, 1]
}

# LightGBM Regressor parameters
lightgbm_params_reg= {
    "n_estimators": [300, 500],
    "learning_rate": [0.01, 0.1],
    "num_leaves": [31, 50, 100],
    "colsample_bytree": [0.7, 1]
}

# Regressor list
regressors = [
    ("CART", DecisionTreeRegressor(), cart_params_reg),
    ("RF", RandomForestRegressor(), rf_params_reg),
    ("XGBoost", XGBRegressor(), xgboost_params_reg),
    ("LightGBM", LGBMRegressor(), lightgbm_params_reg)
]


def hyperparameter_optimization_regression(X_train, y_train, X_test, y_test, regressors, cv=3, scoring="r2"):
    print("Hyperparameter Optimization for Regression...\n")
    best_models = {}

    for name, model, params in regressors:
        print(f"########## {name} ##########")
        
        # Initial score (default parameters)
        cv_score = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
        print(f"{scoring} (CV Before): {cv_score.mean():.4f}")
        
        # GridSearchCV
        gs = GridSearchCV(model, params, cv=cv, n_jobs=-1, verbose=1, scoring=scoring)
        gs.fit(X_train, y_train)
        
        best_model = gs.best_estimator_
        
        # CV score after tuning
        cv_score_after = cross_val_score(best_model, X_train, y_train, cv=cv, scoring=scoring)
        print(f"{scoring} (CV After): {cv_score_after.mean():.4f}")
        
        # Test score
        test_score = best_model.score(X_test, y_test)
        print(f"{scoring} (Test): {test_score:.4f}")
        
        # Best parameters
        print(f"{name} best params: {gs.best_params_}\n")
        best_models[name] = best_model

    return best_models

################################################
# 13. ENSEMBLE LEARNING
################################################
# Stacking & Ensemble Learning
def voting_classifier(best_models, X, y):
    print("Voting Classifier...")
    voting_clf = VotingClassifier(
        estimators=[('CART', best_models["CART"]),
                    ('XGBoost', best_models["XGBoost"]), 
                    ('RF', best_models["RF"]),
                    ('LightGBM', best_models["LightGBM"])],
        voting='soft'
    ).fit(X, y)

    cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
    
    return voting_clf

################################################
# 14. MODEL EVALUATION CURVES
################################################

def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    """
    Function to plot validation curves.
    Can also display None values as labels on the x-axis.
    """
    from sklearn.model_selection import validation_curve
    import numpy as np
    import matplotlib.pyplot as plt

    # Calculate validation curve
    train_score, test_score = validation_curve(
        model,
        X=X,
        y=y,
        param_name=param_name,
        param_range=param_range,
        scoring=scoring,
        cv=cv
    )

    # Mean training and validation scores
    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    # Plotting
    plt.figure(figsize=(8,5))
    plt.plot(mean_train_score, label="Training Score", marker='o', color='blue')
    plt.plot(mean_test_score, label="Validation Score", marker='s', color='green')

    # Use param_range as x-ticks; show "None" if parameter is None
    x_labels = [str(p) if p is not None else "None" for p in param_range]
    plt.xticks(range(len(param_range)), x_labels)

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(param_name)
    plt.ylabel(scoring)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


########################################################################################################
# 15. SMOTE STRATEGY COMPARISON 
def compare_sampling_strategies(X_train, y_train, X_test, y_test, random_state=42):
    """
    Compares 4 strategies:
    1. Baseline (no balancing)
    2. SMOTE
    3. Class Weights
    4. SMOTE + Class Weights

    Returns a DataFrame with weighted F1 scores for each strategy.
    """

    results = []

    # --- 1. Baseline ---
    model_base = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(np.unique(y_train)),
        random_state=random_state,
        n_estimators=500,
        learning_rate=0.05
    )
    model_base.fit(X_train, y_train)
    y_pred_base = model_base.predict(X_test)
    f1_base = f1_score(y_test, y_pred_base, average="weighted")
    results.append({"Strategy": "Baseline", "F1_weighted": f1_base})

    # --- 2. SMOTE ---
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    model_smote = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(np.unique(y_train)),
        random_state=random_state,
        n_estimators=500,
        learning_rate=0.05
    )
    model_smote.fit(X_res, y_res)
    y_pred_smote = model_smote.predict(X_test)
    f1_smote = f1_score(y_test, y_pred_smote, average="weighted")
    results.append({"Strategy": "SMOTE", "F1_weighted": f1_smote})

    # --- 3. Class Weights ---
    model_w = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(np.unique(y_train)),
        random_state=random_state,
        n_estimators=500,
        learning_rate=0.05,
        class_weight="balanced"
    )
    model_w.fit(X_train, y_train)
    y_pred_w = model_w.predict(X_test)
    f1_w = f1_score(y_test, y_pred_w, average="weighted")
    results.append({"Strategy": "Class Weights", "F1_weighted": f1_w})

    # --- 4. SMOTE + Class Weights ---
    model_sw = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(np.unique(y_train)),
        random_state=random_state,
        n_estimators=500,
        learning_rate=0.05,
        class_weight="balanced"
    )
    model_sw.fit(X_res, y_res)
    y_pred_sw = model_sw.predict(X_test)
    f1_sw = f1_score(y_test, y_pred_sw, average="weighted")
    results.append({"Strategy": "SMOTE + Class Weights", "F1_weighted": f1_sw})

    # --- Return results as DataFrame ---
    scores_df = pd.DataFrame(results)

    return scores_df
