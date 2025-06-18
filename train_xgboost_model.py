import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error

def load_and_clean_data(csv_path):
    """
    Loads data from a CSV file, cleans it by replacing '\\N' with NaN,
    and returns the cleaned DataFrame.
    """
    df = pd.read_csv(csv_path)
    df.replace('\\N', np.nan, inplace=True)
    return df

def feature_engineer_part1(df):
    """
    Performs initial feature engineering.
    """
    target_columns = [
        'SCO_WAIT', 'SCO_GTY_MOVE', 'SCO_DOWN_TO_PICK', 'SCO_WAIT_LOCK',
        'SCO_RAISE_TO_PUT', 'SCO_DOWN_TO_PUT', 'SCO_WAIT_DCV', 'SCO_WAIT_END'
    ]
    for col in target_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    categorical_features = [
        'SCO_VSL_CD', 'SCO_VSL_TYPE', 'SCO_STS_ID', 'SCO_WORKFLOW',
        'SCO_WORKTYPE', 'SCO_CNTR1_TYPE', 'SCO_REFSTATUS1', 'SCO_OVLMTCD1',
        'SCO_DTP_DNGGCD1', 'SCO_DH_FG1'
    ]
    for col in categorical_features:
        df[col] = df[col].astype('category')

    numerical_features = ['SCO_VSL_LON', 'SCO_WEIGHT1', 'SCO_CNTR1_SIZE']
    for col in numerical_features:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['SCO_HAS_WI_ID1'] = df['SCO_WI_ID1'].notna().astype(int)
    return df

def feature_engineer_part2(df):
    """
    Performs further feature engineering.
    """
    df['SCO_CNTR1_VLOC_orig_nan'] = df['SCO_CNTR1_VLOC'].isna()
    df['SCO_CNTR1_VLOC'] = df['SCO_CNTR1_VLOC'].astype(str).fillna('000000')
    df['SCO_CNTR1_VLOC_BAY'] = pd.to_numeric(df['SCO_CNTR1_VLOC'].str[0:2], errors='coerce')
    df['SCO_CNTR1_VLOC_STACK'] = pd.to_numeric(df['SCO_CNTR1_VLOC'].str[2:4], errors='coerce')
    df['SCO_CNTR1_VLOC_TIER'] = pd.to_numeric(df['SCO_CNTR1_VLOC'].str[4:6], errors='coerce')
    vloc1_cols = ['SCO_CNTR1_VLOC_BAY', 'SCO_CNTR1_VLOC_STACK', 'SCO_CNTR1_VLOC_TIER']
    df.loc[df['SCO_CNTR1_VLOC_orig_nan'] | (df['SCO_CNTR1_VLOC'] == '000000'), vloc1_cols] = np.nan
    df.drop(columns=['SCO_CNTR1_VLOC_orig_nan'], inplace=True)

    df['IS_DUAL_CONTAINER'] = df['SCO_WI_ID2'].notna().astype(int)
    container2_categorical = ['SCO_CNTR2_TYPE', 'SCO_REFSTATUS2', 'SCO_OVLMTCD2', 'SCO_DTP_DNGGCD2', 'SCO_DH_FG2']
    for col in container2_categorical:
        df[col] = df[col].astype(str).astype('category')
        df.loc[df['IS_DUAL_CONTAINER'] == 0, col] = np.nan
        df[col] = df[col].astype('category')

    container2_numerical = ['SCO_WEIGHT2', 'SCO_CNTR2_SIZE']
    for col in container2_numerical:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df.loc[df['IS_DUAL_CONTAINER'] == 0, col] = np.nan

    df['SCO_CNTR2_VLOC_orig_nan'] = df['SCO_CNTR2_VLOC'].isna()
    df['SCO_CNTR2_VLOC'] = df['SCO_CNTR2_VLOC'].astype(str).fillna('000000')
    df['SCO_CNTR2_VLOC_BAY'] = pd.to_numeric(df['SCO_CNTR2_VLOC'].str[0:2], errors='coerce')
    df['SCO_CNTR2_VLOC_STACK'] = pd.to_numeric(df['SCO_CNTR2_VLOC'].str[2:4], errors='coerce')
    df['SCO_CNTR2_VLOC_TIER'] = pd.to_numeric(df['SCO_CNTR2_VLOC'].str[4:6], errors='coerce')
    vloc2_cols = ['SCO_CNTR2_VLOC_BAY', 'SCO_CNTR2_VLOC_STACK', 'SCO_CNTR2_VLOC_TIER']
    df.loc[df['IS_DUAL_CONTAINER'] == 0 | df['SCO_CNTR2_VLOC_orig_nan'] | (df['SCO_CNTR2_VLOC'] == '000000'), vloc2_cols] = np.nan
    df.drop(columns=['SCO_CNTR2_VLOC_orig_nan'], inplace=True)

    df['SCO_ASTTM_T1'] = pd.to_datetime(df['SCO_ASTTM_T1'], errors='coerce')
    df['SCO_ASTTM_T1_HOUR'] = df['SCO_ASTTM_T1'].dt.hour.astype(float)
    return df

def select_and_drop_features(df):
    """
    Selects specified features, drops others, and returns dataframe and column lists.
    """
    target_columns = [
        'SCO_WAIT', 'SCO_GTY_MOVE', 'SCO_DOWN_TO_PICK', 'SCO_WAIT_LOCK',
        'SCO_RAISE_TO_PUT', 'SCO_DOWN_TO_PUT', 'SCO_WAIT_DCV', 'SCO_WAIT_END'
    ]
    categorical_to_keep = [
        'SCO_VSL_CD', 'SCO_VSL_TYPE', 'SCO_STS_ID', 'SCO_WORKFLOW',
        'SCO_WORKTYPE', 'SCO_CNTR1_TYPE', 'SCO_REFSTATUS1', 'SCO_OVLMTCD1',
        'SCO_DTP_DNGGCD1', 'SCO_DH_FG1', 'SCO_CNTR2_TYPE', 'SCO_REFSTATUS2',
        'SCO_OVLMTCD2', 'SCO_DTP_DNGGCD2', 'SCO_DH_FG2'
    ]
    numerical_to_keep = [
        'SCO_VSL_LON', 'SCO_WEIGHT1', 'SCO_CNTR1_SIZE', 'SCO_HAS_WI_ID1',
        'SCO_CNTR1_VLOC_BAY', 'SCO_CNTR1_VLOC_STACK', 'SCO_CNTR1_VLOC_TIER',
        'IS_DUAL_CONTAINER', 'SCO_WEIGHT2', 'SCO_CNTR2_SIZE',
        'SCO_CNTR2_VLOC_BAY', 'SCO_CNTR2_VLOC_STACK', 'SCO_CNTR2_VLOC_TIER',
        'SCO_ASTTM_T1_HOUR'
    ]
    all_columns_to_keep = target_columns + categorical_to_keep + numerical_to_keep

    missing_cols = [col for col in all_columns_to_keep if col not in df.columns]
    if missing_cols:
        print(f"Warning: Columns to keep but missing from DF: {missing_cols}")
        all_columns_to_keep = [col for col in all_columns_to_keep if col in df.columns]
        numerical_to_keep = [col for col in numerical_to_keep if col in all_columns_to_keep]
        categorical_to_keep = [col for col in categorical_to_keep if col in all_columns_to_keep]
        target_columns = [col for col in target_columns if col in all_columns_to_keep]

    df_selected = df[all_columns_to_keep].copy()

    final_numerical_kept = [col for col in numerical_to_keep if col in df_selected.columns]
    final_categorical_kept = [col for col in categorical_to_keep if col in df_selected.columns]
    final_target_columns = [col for col in target_columns if col in df_selected.columns]

    return df_selected, final_numerical_kept, final_categorical_kept, final_target_columns

def handle_missing_values(df, numerical_cols, categorical_cols):
    """
    Handles missing values in the DataFrame.
    """
    for col in numerical_cols:
        if col in df.columns:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
        else:
            print(f"Warning: Numerical column {col} for imputation not found in DataFrame.")

    for col in categorical_cols:
        if col in df.columns:
            if isinstance(df[col].dtype, pd.CategoricalDtype): # Check using isinstance
                 if "Missing" not in df[col].cat.categories:
                    df[col] = df[col].cat.add_categories("Missing")
            else:
                df[col] = df[col].astype('category')
                if "Missing" not in df[col].cat.categories:
                    df[col] = df[col].cat.add_categories("Missing")
            df[col].fillna("Missing", inplace=True)
        else:
            print(f"Warning: Categorical column {col} for imputation not found in DataFrame.")

    return df

def train_model(df, feature_cols, target_cols_list):
    """
    Trains a MultiOutput XGBoost model.
    """
    X = df[feature_cols]
    y = df[target_cols_list]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    xgb_estimator = xgb.XGBRegressor(
        objective='reg:squarederror',
        enable_categorical=True,
        random_state=42
    )

    multioutput_model = MultiOutputRegressor(xgb_estimator)
    multioutput_model.fit(X_train, y_train)

    print("Model training completed.")
    return multioutput_model, X_test, y_test

def evaluate_model(model, X_test, y_test, target_cols_list_eval):
    """
    Evaluates the model and prints MSE for each target.
    """
    y_pred = model.predict(X_test)

    print("\nModel Evaluation (Mean Squared Error for each target):")
    for i, col_name in enumerate(target_cols_list_eval):
        mse = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
        print(f"{col_name}: {mse:.4f}")

if __name__ == '__main__':
    # Load and clean
    cleaned_df = load_and_clean_data('sch_sts_oee.csv')
    print("First 5 rows of cleaned data:")
    print(cleaned_df.head())
    print("\n" + "="*50 + "\n")

    # Feature engineering part 1
    df_featured_p1 = feature_engineer_part1(cleaned_df.copy())
    print("DataFrame info after feature engineering (part 1):")
    df_featured_p1.info()
    print("\n" + "="*50 + "\n")

    # Feature engineering part 2
    df_featured_p2 = feature_engineer_part2(df_featured_p1.copy())
    print("DataFrame info after feature engineering (part 2):")
    df_featured_p2.info()
    print("\nFirst 5 rows after feature engineering (part 2):")
    print(df_featured_p2.head())
    print("\n" + "="*50 + "\n")

    # Select features
    df_selected, numerical_cols_kept, categorical_cols_kept, target_cols_list_main = select_and_drop_features(df_featured_p2)
    print("DataFrame info after selecting and dropping features:")
    df_selected.info()
    print("\nFirst 5 rows after selecting and dropping features:")
    print(df_selected.head())
    print("\n" + "="*50 + "\n")

    all_numerical_for_imputation = list(set(numerical_cols_kept + target_cols_list_main))

    df_imputed = handle_missing_values(df_selected.copy(), all_numerical_for_imputation, categorical_cols_kept)
    print("DataFrame info after imputing missing values:")
    df_imputed.info()
    print(f"\nTotal NaNs in DataFrame after imputation: {df_imputed.isnull().sum().sum()}")
    print("\nFirst 5 rows after imputing missing values:")
    print(df_imputed.head())
    print("\n" + "="*50 + "\n")

    # Train model
    feature_columns = numerical_cols_kept + categorical_cols_kept
    trained_model_obj, X_test_data, y_test_data = train_model(df_imputed, feature_columns, target_cols_list_main)

    # Evaluate model
    evaluate_model(trained_model_obj, X_test_data, y_test_data, target_cols_list_main)
