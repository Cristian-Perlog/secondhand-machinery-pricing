import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from category_encoders import TargetEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
import shap
import logging
import warnings
warnings.filterwarnings('ignore')

# Configuration
class Config:
    SAMPLE_SIZE = 25000 # Used data sampling -> https://www.geeksforgeeks.org/data-analysis/handling-large-datasets-efficiently-on-non-super-computers/
    TEST_SIZE = 0.2 # https://developers.google.com/machine-learning/crash-course/overfitting/dividing-datasets
    VAL_SIZE = 0.25 # Same source as above
    RANDOM_STATE = 42
    N_JOBS = min(4, os.cpu_count() - 1) if os.cpu_count() else 1
    HIGH_CARDINALITY_THRESHOLD = 5 # Seems like a reasonable value
    MISSING_VALUE_THRESHOLD = 0.7 # https://suparnachowdhury.medium.com/the-art-of-feature-engineering-handling-missing-values-fd6b4290a99e

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ML Pipeline")

def load_data(filepath):
    """Load and sample data"""
    try:
        df_full = pd.read_csv(filepath, low_memory=False)
        df_sample = df_full.sample(n=min(Config.SAMPLE_SIZE, len(df_full)), random_state=Config.RANDOM_STATE)
        logger.info(f"Loaded random sample of {len(df_sample)} rows from {len(df_full)} total")
        return df_sample
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

def analyze_columns(df):
    """Print detailed column information for feature selection"""
    print("\n" + "="*80)
    print("COLUMN ANALYSIS FOR FEATURE SELECTION")
    print("="*80 + "\n")
    
    # Create a safe version of min/max/mean/median that handles mixed types
    def safe_stats(series):
        try:
            numeric_series = pd.to_numeric(series, errors='coerce')
            return {
                'Min': numeric_series.min(),
                'Max': numeric_series.max(),
                'Mean': numeric_series.mean(),
                'Median': numeric_series.median()
            }
        except:
            return {
                'Min': None,
                'Max': None,
                'Mean': None,
                'Median': None
            }
    
    # Build column info
    column_info = []
    for col in df.columns:
        stats = safe_stats(df[col])
        col_info = {
            'Column': col,
            'Data Type': df[col].dtype,
            'Unique Values': df[col].nunique(),
            'Missing Values': df[col].isnull().sum(),
            'Missing %': round(df[col].isnull().mean() * 100, 2),
            'Min': stats['Min'],
            'Max': stats['Max'],
            'Mean': stats['Mean'],
            'Median': stats['Median']
        }
        column_info.append(col_info)
    
    column_df = pd.DataFrame(column_info)
    
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', '{:.2f}'.format)
    
    print(column_df)

def preprocess_data(df, target_col):
    """Handle missing values and basic cleaning"""
    logger.info("Starting data preprocessing")
    
    # Drop identificators columns
    id_cols_to_drop = [
        'Unnamed: 0', 'Sales ID', 'Machine ID', 'Model ID', 'Auctioneer ID', 'datasource'
    ]
    df = df.drop(id_cols_to_drop, axis=1)
    logger.info(f"Dropped {len(id_cols_to_drop)} columns with identificator values")

    # Drop high-missing columns
    missing = df.isnull().mean()
    cols_to_drop = missing[missing > Config.MISSING_VALUE_THRESHOLD].index
    df = df.drop(cols_to_drop, axis=1)
    logger.info(f"Dropped {len(cols_to_drop)} columns with missing values > {Config.MISSING_VALUE_THRESHOLD}")

    # Split sales date column in year, month and day and then drop the original column
    df['Sales year'] = pd.to_datetime(df['Sales date']).dt.year
    df['Sales month'] = pd.to_datetime(df['Sales date']).dt.month
    df['Sales day'] = pd.to_datetime(df['Sales date']).dt.day
    df = df.drop('Sales date', axis=1)

    # Create a new feature to determine how old a machine is (Might be useful)
    df['Machine age'] = df['Sales year'] - df['Year Made']

    # Drop rows with missing target (should be none such rows)
    if df[target_col].isnull().any():
        df = df.dropna(subset=[target_col])
        logger.info(f"Dropped rows with missing target values. Shape: {df.shape}")
    
    # Fill remaining missing values -> https://www.kaggle.com/code/abhayparashar31/feature-engineering-handling-missing-values
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('Unknown')
            else:
                df[col] = df[col].fillna(df[col].median())
    
    return df

# https://www.geeksforgeeks.org/machine-learning/categorical-data-encoding-techniques-in-machine-learning/
def encode_categoricals(df, target_col):
    """Handle categorical variable encoding"""
    logger.info("Starting categorical encoding")
    
    # Get categorical columns
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    if target_col in cat_cols:
        cat_cols.remove(target_col)
    
    if not cat_cols:
        logger.info("No categorical columns found")
        return df
    
    logger.info(f"Found {len(cat_cols)} categorical columns: {cat_cols}")
    
    # Separate high and low cardinality columns
    high_card_cols = [col for col in cat_cols if df[col].nunique() > Config.HIGH_CARDINALITY_THRESHOLD]
    low_card_cols = [col for col in cat_cols if df[col].nunique() <= Config.HIGH_CARDINALITY_THRESHOLD]
    
    # Encode low cardinality columns with LabelEncoder
    if low_card_cols:
        logger.info(f"Encoding {len(low_card_cols)} low cardinality columns with LabelEncoder")
        for col in low_card_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    
    # Encode high cardinality columns with TargetEncoder
    if high_card_cols:
        logger.info(f"Encoding {len(high_card_cols)} high cardinality columns with TargetEncoder")
        target_encoder = TargetEncoder(cols=high_card_cols, smoothing=1.0)
        df[high_card_cols] = target_encoder.fit_transform(df[high_card_cols], df[target_col])
    
    # Verify no object columns remain
    remaining_objects = df.select_dtypes(include='object').columns.tolist()
    if target_col in remaining_objects:
        remaining_objects.remove(target_col)
    
    if remaining_objects:
        logger.warning(f"Still have object columns: {remaining_objects}")
        # Force convert any remaining object columns
        for col in remaining_objects:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())
    
    logger.info("Categorical encoding completed")
    return df

def create_visualizations(df, target_col):
    """Generate exploratory data analysis visualizations"""
    logger.info("Creating visualizations")
    
    # Histogram of target
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    sns.histplot(df[target_col].dropna(), kde=True)
    plt.title(f'Distribution of {target_col}')
        
    # Correlation with target
    plt.subplot(2, 2, 3)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = df[numeric_cols].corr()[target_col].sort_values(ascending=False)
    top_corr = correlations.head(10)
    sns.barplot(x=top_corr.values, y=top_corr.index)
    plt.title('Top 10 Correlations with Target')
        
    plt.tight_layout()
    plt.savefig("eda_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    corr = df.select_dtypes(include=[np.number]).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return df

# Choose best metrics for regression models -> https://developer.nvidia.com/blog/a-comprehensive-overview-of-regression-evaluation-metrics/ 
def evaluate_model(model, X_val, y_val, X_test, y_test, model_name):
    """Comprehensive model evaluation"""
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    
    results = {
        'validation': {
            'mse': mean_squared_error(y_val, val_pred),
            'rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
            'mae': mean_absolute_error(y_val, val_pred),
            'r2': r2_score(y_val, val_pred)
        },
        'test': {
            'mse': mean_squared_error(y_test, test_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'mae': mean_absolute_error(y_test, test_pred),
            'r2': r2_score(y_test, test_pred)
        }
    }
    
    # Plot actual vs predicted
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_val, val_pred, alpha=0.5, label='Validation')
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{model_name}: Validation Set")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, test_pred, alpha=0.5, label='Test')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{model_name}: Test Set")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{model_name.lower().replace(' ', '_')}_predictions.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return results


# Training of Linear Models (Standard Linear, Ridge, Lasso)
# Good baseline models that are interpretable and work well when relationships are linear or nearly linear
# So not the best for this problem, but still nice for evaluation and comparison purposes
# Ridge/Lasso help prevent overfitting through different regularization techniques
def train_linear_models(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train multiple linear models with regularization"""
    logger.info("Training Linear Models")
    
    models = {}
    results = {}
    
    # Standard Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models['Linear Regression'] = lr
    results['Linear Regression'] = evaluate_model(lr, X_val, y_val, X_test, y_test, 'Linear Regression')
    
    # Ridge Regression
    ridge_params = {'alpha': [0.1, 1.0, 10.0, 100.0]}
    ridge = Ridge(random_state=Config.RANDOM_STATE)
    ridge_grid = GridSearchCV(ridge, ridge_params, cv=3, scoring='r2', n_jobs=Config.N_JOBS)
    ridge_grid.fit(X_train, y_train)
    models['Ridge Regression'] = ridge_grid.best_estimator_
    results['Ridge Regression'] = evaluate_model(ridge_grid.best_estimator_, X_val, y_val, X_test, y_test, 'Ridge Regression')
    results['Ridge Regression']['best_params'] = ridge_grid.best_params_
    
    # Lasso Regression
    lasso_params = {'alpha': [0.1, 1.0, 10.0, 100.0]}
    lasso = Lasso(random_state=Config.RANDOM_STATE, max_iter=2000)
    lasso_grid = GridSearchCV(lasso, lasso_params, cv=3, scoring='r2', n_jobs=Config.N_JOBS)
    lasso_grid.fit(X_train, y_train)
    models['Lasso Regression'] = lasso_grid.best_estimator_
    results['Lasso Regression'] = evaluate_model(lasso_grid.best_estimator_, X_val, y_val, X_test, y_test, 'Lasso Regression')
    results['Lasso Regression']['best_params'] = lasso_grid.best_params_
    
    return models, results

# Training of Random Forest Model
# Handles non-linear relationships well, robust to outliers and noise, thus better than linear models
# Provides feature importance scores which could help with interpretability

def train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train and evaluate Random Forest with hyperparameter tuning"""
    logger.info("Training Random Forest model")
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    rf = RandomForestRegressor(random_state=Config.RANDOM_STATE, n_jobs=Config.N_JOBS)
    grid_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,
        n_iter=20,
        cv=3,
        scoring='r2',
        n_jobs=Config.N_JOBS,
        verbose=1,
        random_state=Config.RANDOM_STATE
    )
    
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_
    
    results = evaluate_model(best_rf, X_val, y_val, X_test, y_test, 'Random Forest')
    results['best_params'] = grid_search.best_params_
    
    # Plot feature importances
    importances = best_rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title("Random Forest Feature Importances")
    top_features = min(20, len(indices))
    plt.bar(range(top_features), importances[indices[:top_features]])
    plt.xticks(range(top_features), X_train.columns[indices[:top_features]], rotation=45)
    plt.tight_layout()
    plt.savefig("random_forest_feature_importance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return best_rf, results

# Training of Gradient Boosting Model
# Sequentially corrects errors from previous trees, often achieving high accuracy
# Handles mixed data types well and can capture complex patterns, thus should be one of the better models for this problem

def train_gradient_boosting(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train Gradient Boosting model"""
    logger.info("Training Gradient Boosting model")
    
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9],
        'max_features': ['sqrt', 'log2']
    }
    
    gb = GradientBoostingRegressor(random_state=Config.RANDOM_STATE)
    grid_search = RandomizedSearchCV(
        gb, param_grid, n_iter=15, cv=3, scoring='r2', 
        n_jobs=Config.N_JOBS, verbose=1, random_state=Config.RANDOM_STATE
    )
    
    grid_search.fit(X_train, y_train)
    best_gb = grid_search.best_estimator_
    
    results = evaluate_model(best_gb, X_val, y_val, X_test, y_test, 'Gradient Boosting')
    results['best_params'] = grid_search.best_params_
    
    return best_gb, results

# Training of Extreme Gradient Boosting Model
# Provides parallel tree boosting and is the leading machine learning library for regression problems
# Advantages over Gradient Boosting include regularization techniques and parallel processing to enhance speed and reduce overfitting
def train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train and evaluate XGBoost model with hyperparameter tuning"""
    logger.info("Training XGBoost model")
    
    # Ensure all columns are numeric
    X_train = X_train.select_dtypes(include=[np.number])
    X_val = X_val[X_train.columns]
    X_test = X_test[X_train.columns]
    
    # Convert to float32 to save memory
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    X_test = X_test.astype(np.float32)
    
    param_dist = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [1.0, 1.5, 2.0]
    }

    xgb = XGBRegressor(
        objective='reg:squarederror',
        random_state=Config.RANDOM_STATE,
        n_jobs=Config.N_JOBS,
        tree_method='hist'
    )
    
    random_search = RandomizedSearchCV(
        xgb,
        param_dist,
        n_iter=30,
        cv=3,
        scoring='r2',
        n_jobs=1,
        verbose=1,
        random_state=Config.RANDOM_STATE
    )
    
    random_search.fit(X_train, y_train)
    best_xgb = random_search.best_estimator_
    
    results = evaluate_model(best_xgb, X_val, y_val, X_test, y_test, 'XGBoost')
    results['best_params'] = random_search.best_params_

    return best_xgb, results

def compare_models(results):
    """Create model comparison visualization"""
    logger.info("Creating model comparison")
    
    models = list(results.keys())
    test_r2 = [results[model]['test']['r2'] for model in models]
    test_rmse = [results[model]['test']['rmse'] for model in models]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # R² comparison
    bars1 = ax1.bar(models, test_r2, color='skyblue', edgecolor='navy', alpha=0.7)
    ax1.set_title('Model Comparison - R² Score')
    ax1.set_ylabel('R² Score')
    ax1.set_ylim(0, 1)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, value in zip(bars1, test_r2):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    # RMSE comparison
    bars2 = ax2.bar(models, test_rmse, color='lightcoral', edgecolor='darkred', alpha=0.7)
    ax2.set_title('Model Comparison - RMSE')
    ax2.set_ylabel('RMSE')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, value in zip(bars2, test_rmse):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(test_rmse)*0.01, 
                f'{value:.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    try:
        # 1. Load and prepare data
        df = load_data('Secondhand_Machinery_Data.csv')
        if df is None:
            return

        # Analyze columns for feature engineering
        analyze_columns(df)
        
        # Identifying target column - based on the columns analysis
        target_col = None
        for col in df.columns:
            if 'sales price' in col.lower():
                target_col = col
                break
        
        if target_col is None:
            logger.error("Target column not found")
            return
        
        logger.info(f"Target column identified: {target_col}")
                
        # 2. Preprocessing pipeline
        df = preprocess_data(df, target_col)
        df = encode_categoricals(df, target_col)
        df = create_visualizations(df, target_col)
        
        # 3. Prepare features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Remove any remaining non-numeric columns
        X = X.select_dtypes(include=[np.number])
        
        logger.info(f"Final feature shape: {X.shape}")
        logger.info(f"Target shape: {y.shape}")
        
        # 4. Train-test split
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=Config.VAL_SIZE, random_state=Config.RANDOM_STATE
        )
        
        logger.info(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}, Test shape: {X_test.shape}")
        
        # 5. Model training and evaluation
        all_models = {}
        all_results = {}
        
        # Linear Models
        linear_models, linear_results = train_linear_models(X_train, y_train, X_val, y_val, X_test, y_test)
        all_models.update(linear_models)
        all_results.update(linear_results)
        
        # Random Forest
        rf_model, rf_results = train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test)
        all_models['Random Forest'] = rf_model
        all_results['Random Forest'] = rf_results
        
        # Gradient Boosting
        gb_model, gb_results = train_gradient_boosting(X_train, y_train, X_val, y_val, X_test, y_test)
        all_models['Gradient Boosting'] = gb_model
        all_results['Gradient Boosting'] = gb_results
        
        # XGBoost
        xgb_model, xgb_results = train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test)
        all_models['XGBoost'] = xgb_model
        all_results['XGBoost'] = xgb_results
        
        # 6. Model comparison
        compare_models(all_results)
        
        # 7. Print final results
        logger.info("\n" + "="*50)
        logger.info("FINAL MODEL COMPARISON")
        
        # Sort models by test R²
        sorted_models = sorted(all_results.items(), key=lambda x: x[1]['test']['r2'], reverse=True)
        
        for i, (model_name, model_results) in enumerate(sorted_models, 1):
            logger.info(f"\n{i}. {model_name.upper()}")
            logger.info("-" * 30)
            if 'best_params' in model_results:
                logger.info(f"Best parameters: {model_results['best_params']}")
            logger.info(f"Validation R²: {model_results['validation']['r2']:.4f}")
            logger.info(f"Test R²: {model_results['test']['r2']:.4f}")
            logger.info(f"Test RMSE: {model_results['test']['rmse']:.2f}")
            logger.info(f"Test MAE: {model_results['test']['mae']:.2f}")
        
        # Best model summary
        best_model_name = sorted_models[0][0]
        best_model_results = sorted_models[0][1]
        logger.info(f"\n🏆 BEST MODEL: {best_model_name}")
        logger.info(f"Test R²: {best_model_results['test']['r2']:.4f}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)

if __name__ == "__main__":
    main()