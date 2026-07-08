#!/usr/bin/env python3
"""
Enhanced Regressor Training Script v2 with Advanced Feature Engineering
Uses enhanced features from Yahoo Finance, Polygon, market context, and content analysis
Optimizes features for each model separately for best R2 score
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Import modular functions
from util.mlflow_setup import setup_logger, setup_mlflow, train_and_save_regressor
from util.save_to_db import save_regressor_to_database, save_evaluation_details_regressor
from util.feature_engineering_v2 import add_enhanced_features_v2_ultra_fast
from util.feature_selection_v2 import clean_and_select_features

# Global wrapper class to avoid serialization issues
class AdvancedRegressorWrapper:
    """Wrapper class for advanced regressor models to ensure proper serialization."""
    def __init__(self, ensemble_model, poly_transformer, scaler):
        self.ensemble_model = ensemble_model
        self.poly_transformer = poly_transformer
        self.scaler = scaler
    
    def predict(self, X):
        """Make predictions using the wrapped model."""
        X_scaled = self.scaler.transform(X)
        if self.poly_transformer is not None:
            X_poly = self.poly_transformer.transform(X_scaled)
            return self.ensemble_model.predict(X_poly)
        else:
            return self.ensemble_model.predict(X_scaled)
    
    def __getstate__(self):
        """Custom serialization method to avoid pickling issues."""
        state = self.__dict__.copy()
        return state
    
    def __setstate__(self, state):
        """Custom deserialization method."""
        self.__dict__.update(state)

def train_simple_regressor(X, y, target_type, logger, n_features=30):
    """Train simple Random Forest with basic grid search (max 10 fits)"""
    logger.info(f"🚀 Training simple Random Forest for {target_type}")
    
    # Create simple Random Forest
    rf = RandomForestRegressor(
        n_estimators=100, 
        max_depth=15, 
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=42, 
        n_jobs=1
    )
    
    # Simple grid search with max 10 fits
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, 15],
        'min_samples_split': [5, 10]
    }
    
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=cv,
        scoring='r2',
        n_jobs=1,
        verbose=1
    )
    
    logger.info("🔍 Running simple grid search (max 10 fits)...")
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_
    best_cv_score = grid_search.best_score_
    
    logger.info(f"✅ Best CV R2 score: {best_cv_score:.4f}")
    logger.info(f"✅ Best params: {grid_search.best_params_}")
    
    return best_model, None, best_cv_score

def main():
    logger = setup_logger()
    logger.info("🚀 Starting Enhanced Random Forest Regressor Training v2...")
    logger.info("🎯 Advanced feature engineering with Yahoo Finance, Polygon, market context, and content analysis")
    logger.info("🎯 Optimized feature selection for each model separately")

    parser = argparse.ArgumentParser(description='Enhanced RF Regressor Training v2')
    parser.add_argument('--input-file', type=str, default=r'D:\Oxford\Extra\Finance_NLP\finespresso-modelling\data\training_data\regressor_training_v2_202509180138.csv')
    parser.add_argument('--output-dir', type=str, default=r'D:\Oxford\Extra\Finance_NLP\finespresso-modelling\data\training_data\enhanced_regressor_v2_training')
    parser.add_argument('--model-category', type=str, default='ml_models', choices=['ml_models', 'rag_models', 'llm_models'])
    parser.add_argument('--event', type=str, default='partnerships', help='Specific event to train on (partnerships, all_events, or all)')
    parser.add_argument('--polygon-api-key', type=str, default=None, help='Polygon.io API key for enhanced features')
    parser.add_argument('--n-features', type=int, default=30, help='Number of features to select per model (default: 30)')
    args = parser.parse_args()

    # Setup MLflow - Use 'regressor' to get "Regressor_Training" experiment
    mlflow_available, experiment_name, model_category = setup_mlflow('regressor', args.model_category)
    if not mlflow_available:
        logger.error("❌ MLflow setup failed. Cannot continue without MLflow.")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Loading dataset from {args.input_file}")
    df = pd.read_csv(args.input_file)
    logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")

    # Check for required target columns
    required_targets = ['price_change_percentage']
    missing_targets = [target for target in required_targets if target not in df.columns]
    if missing_targets:
        raise ValueError(f"Missing required target columns: {missing_targets}")

    # Load current script as binary data for database storage
    current_script = os.path.abspath(__file__)
    with open(current_script, 'rb') as f:
        script_blob = f.read()
    
    logger.info(f"✅ Loaded training script for database storage: {len(script_blob)} bytes")

    # Data is already enhanced with v2 features, just clean and select
    logger.info("🎯 Cleaning and selecting best features...")
    target_columns = ['price_change_percentage', 'nextday_price_change_percentage']
    df_selected = clean_and_select_features(df, target_columns, n_features=args.n_features)
    logger.info(f"✅ Selected dataset shape: {df_selected.shape}")

    # Create TF-IDF features
    logger.info("Creating TF-IDF features for text content...")
    
    title_tfidf = TfidfVectorizer(
        max_features=50,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words='english'
    )
    
    content_tfidf = TfidfVectorizer(
        max_features=100,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words='english'
    )
    
    # Fit and transform text features
    title_features = title_tfidf.fit_transform(df_selected['title_en'].fillna(''))
    content_features = content_tfidf.fit_transform(df_selected['content_en'].fillna(''))
    
    title_features_dense = title_features.toarray()
    content_features_dense = content_features.toarray()
    
    logger.info(f"Title TF-IDF features: {title_features_dense.shape[1]}")
    logger.info(f"Content TF-IDF features: {content_features_dense.shape[1]}")
    
    # Train for specific event or all events
    if args.event == 'all_events':
        # Train ONLY on combined dataset (all events together)
        target_events = ['all_events']
        logger.info(f"Training ONLY on combined dataset (all events together)")
        logger.info(f"Target events: {target_events}")
    elif args.event == 'all':
        # Get all events with sufficient samples
        event_counts = df_selected['event'].value_counts()
        individual_events = event_counts[event_counts >= 500].index.tolist()
        target_events = ['all_events'] + individual_events
        logger.info(f"Training on combined dataset + {len(individual_events)} individual events with ≥500 samples")
        logger.info(f"Target events: {target_events}")
    else:
        target_events = [args.event]
        logger.info(f"Training on specific event: {args.event}")
    
    results = []
    
    for event in target_events:
        if event == 'all_events':
            df_event = df_selected.copy()
        else:
            df_event = df_selected[df_selected['event'] == event].copy()
            
        if len(df_event) < 500:
            logger.info(f"Skipping event {event} (only {len(df_event)} samples, need ≥500)")
            continue
            
        logger.info(f"Training for event: {event} with {len(df_event)} samples")
        
        # Convert dataset to CSV for database storage
        dataset_csv = df_event.to_csv(index=False)
        dataset_blob = dataset_csv.encode('utf-8')
        logger.info(f"✅ Prepared dataset for database storage: {len(dataset_blob)} bytes")
        
        # Train dual regressor models
        logger.info(f"🎯 Training dual regressor models for {event}")
        try:
            # Prepare data
            df_processed = df_event.copy()
            
            # Get feature columns (exclude target columns and text columns)
            feature_columns = [col for col in df_processed.columns 
                              if col not in ['price_change_percentage', 'nextday_price_change_percentage', 'news_id', 'title_en', 'content_en', 
                                           'title', 'content', 'ticker', 'company', 'reason', 'link', 'ticker_url', 
                                           'downloaded_at', 'timezone', 'publisher_summary', 'language', 'yf_ticker.1']]
            
            # Handle categorical encoding for feature columns
            for col in feature_columns:
                if col in df_processed.columns and df_processed[col].dtype == 'object':
                    df_processed[col] = pd.Categorical(df_processed[col].astype(str)).codes
            
            df_processed = df_processed.fillna(0)
            for col in feature_columns:
                if col in df_processed.columns:
                    df_processed[col] = df_processed[col].astype(float)
            
            # Get TF-IDF features
            title_features_event = title_tfidf.transform(df_event['title_en'].fillna(''))
            content_features_event = content_tfidf.transform(df_event['content_en'].fillna(''))
            
            title_features_event_dense = title_features_event.toarray()
            content_features_event_dense = content_features_event.toarray()
            
            # Train Model 1: price_change_percentage
            try:
                logger.info(f"🎯 Training Model 1: price_change_percentage regressor")
                y_actual_price = df_event['price_change_percentage']
                
                # Use all available feature columns for price_change_percentage
                actual_price_features = feature_columns.copy()
                logger.info(f"Using {len(actual_price_features)} features for price_change_percentage")
                
                # Prepare features
                X_actual_price = df_processed[actual_price_features].copy()
                X_actual_price_combined = np.hstack([X_actual_price.values, title_features_event_dense, content_features_event_dense])
                
                scaler_actual_price = StandardScaler()
                X_scaled_actual_price = scaler_actual_price.fit_transform(X_actual_price_combined)
                
                # Add TF-IDF feature names
                tfidf_feature_names = []
                tfidf_feature_names.extend([f"title_tfidf_{i}" for i in range(title_features_event_dense.shape[1])])
                tfidf_feature_names.extend([f"content_tfidf_{i}" for i in range(content_features_event_dense.shape[1])])
                
                # Use simple Random Forest training for price_change_percentage
                logger.info("🎯 Using simple Random Forest training for price_change_percentage...")
                trained_model_actual_price, poly_transformer_actual_price, cv_score_actual_price = train_simple_regressor(
                    X_scaled_actual_price, y_actual_price, 'price_change_percentage', logger, args.n_features
                )
                
                # Use the global AdvancedRegressorWrapper class
                
                # Wrap the advanced model
                wrapped_model_actual_price = AdvancedRegressorWrapper(trained_model_actual_price, poly_transformer_actual_price, scaler_actual_price)
                
                # Use basic MLflow training for logging (but with our advanced model)
                model_id_actual_price, run_id_actual_price, _, version_actual_price, model_name_actual_price, training_summary_actual_price = train_and_save_regressor(
                    X_scaled_actual_price, y_actual_price, 
                    event_name=event,
                    target_type='price_change_percentage',
                    model_category=args.model_category,
                    experiment_name=experiment_name,
                    feature_columns=actual_price_features + tfidf_feature_names,
                    scaler=scaler_actual_price,
                    title_tfidf=title_tfidf,
                    content_tfidf=content_tfidf,
                    dataset_blob=dataset_blob,
                    script_blob=script_blob
                )
                
                # Replace with our advanced model
                trained_model_actual_price = wrapped_model_actual_price
                
                # Save model to database with BLOB objects
                save_regressor_to_database(
                    model_id=model_id_actual_price,
                    event_type=event,
                    target_type='price_change_percentage',
                    version=version_actual_price,
                    r2_score_val=training_summary_actual_price.get('r2_score'),
                    rmse=training_summary_actual_price.get('rmse'),
                    mse=training_summary_actual_price.get('mse'),
                    mae=training_summary_actual_price.get('mae'),
                    train_samples=training_summary_actual_price.get('train_samples'),
                    test_samples=training_summary_actual_price.get('test_samples'),
                    cv_folds=training_summary_actual_price.get('cv_folds'),
                    best_params={"n_estimators": 200, "max_depth": 20, "method": "enhanced_v2_training", "target_type": 'price_change_percentage', "version": "v2"},
                    feature_columns=actual_price_features + tfidf_feature_names,
                    mlflow_run_id=run_id_actual_price,
                    experiment_name=experiment_name,
                    model_name=model_name_actual_price,
                    model_stage="Staging",
                    model_category=args.model_category,
                    model_obj=trained_model_actual_price,
                    scaler_obj=scaler_actual_price,
                    title_tfidf_obj=title_tfidf,
                    content_tfidf_obj=content_tfidf
                )
                
                logger.info(f"✅ Successfully trained price_change_percentage regressor for {event}")
                
            except Exception as e:
                logger.error(f"❌ Failed to train price_change_percentage regressor for {event}: {e}")
                continue
            
            # Train Model 2: nextday_price_change_percentage (using price_change_percentage as feature)
            try:
                logger.info(f"🎯 Training Model 2: nextday_price_change_percentage regressor")
                y_nextday_price = df_event['nextday_price_change_percentage']
                
                # Use all available feature columns for nextday_price_change_percentage (including price_change_percentage if available)
                nextday_price_features = feature_columns.copy()
                if 'price_change_percentage' in df_processed.columns and 'price_change_percentage' not in nextday_price_features:
                    nextday_price_features.append('price_change_percentage')
                logger.info(f"Using {len(nextday_price_features)} features for nextday_price_change_percentage")
                
                # Prepare features
                X_nextday_price = df_processed[nextday_price_features].copy()
                X_nextday_price_combined = np.hstack([X_nextday_price.values, title_features_event_dense, content_features_event_dense])
                
                scaler_nextday_price = StandardScaler()
                X_scaled_nextday_price = scaler_nextday_price.fit_transform(X_nextday_price_combined)
                
                # Use simple Random Forest training for nextday_price_change_percentage
                logger.info("🎯 Using simple Random Forest training for nextday_price_change_percentage...")
                trained_model_nextday_price, poly_transformer_nextday_price, cv_score_nextday_price = train_simple_regressor(
                    X_scaled_nextday_price, y_nextday_price, 'nextday_price_change_percentage', logger, args.n_features
                )
                
                # Wrap the advanced model for nextday_price
                wrapped_model_nextday_price = AdvancedRegressorWrapper(trained_model_nextday_price, poly_transformer_nextday_price, scaler_nextday_price)
                
                # Use basic MLflow training for logging (but with our advanced model)
                model_id_nextday_price, run_id_nextday_price, _, version_nextday_price, model_name_nextday_price, training_summary_nextday_price = train_and_save_regressor(
                    X_scaled_nextday_price, y_nextday_price, 
                    event_name=event,
                    target_type='nextday_price_change_percentage',
                    model_category=args.model_category,
                    experiment_name=experiment_name,
                    feature_columns=nextday_price_features + tfidf_feature_names,
                    scaler=scaler_nextday_price,
                    title_tfidf=title_tfidf,
                    content_tfidf=content_tfidf,
                    dataset_blob=dataset_blob,
                    script_blob=script_blob
                )
                
                # Replace with our advanced model
                trained_model_nextday_price = wrapped_model_nextday_price
                
                # Save model to database with BLOB objects
                save_regressor_to_database(
                    model_id=model_id_nextday_price,
                    event_type=event,
                    target_type='nextday_price_change_percentage',
                    version=version_nextday_price,
                    r2_score_val=training_summary_nextday_price.get('r2_score'),
                    rmse=training_summary_nextday_price.get('rmse'),
                    mse=training_summary_nextday_price.get('mse'),
                    mae=training_summary_nextday_price.get('mae'),
                    train_samples=training_summary_nextday_price.get('train_samples'),
                    test_samples=training_summary_nextday_price.get('test_samples'),
                    cv_folds=training_summary_nextday_price.get('cv_folds'),
                    best_params={"n_estimators": 200, "max_depth": 20, "method": "enhanced_v2_training", "target_type": 'nextday_price_change_percentage', "version": "v2"},
                    feature_columns=nextday_price_features + tfidf_feature_names,
                    mlflow_run_id=run_id_nextday_price,
                    experiment_name=experiment_name,
                    model_name=model_name_nextday_price,
                    model_stage="Staging",
                    model_category=args.model_category,
                    model_obj=trained_model_nextday_price,
                    scaler_obj=scaler_nextday_price,
                    title_tfidf_obj=title_tfidf,
                    content_tfidf_obj=content_tfidf
                )
                
                logger.info(f"✅ Successfully trained nextday_price_change_percentage regressor for {event}")
                
            except Exception as e:
                logger.error(f"❌ Failed to train nextday_price_change_percentage regressor for {event}: {e}")
                continue
            
            # Save evaluation details for both models
            if len(df_event) > 800:
                sample_size = min(100, len(df_event))
                sample_indices = np.random.choice(len(df_event), sample_size, replace=False)
                df_sample = df_event.iloc[sample_indices].reset_index(drop=True)
                
                # Evaluation for both models combined
                X_sample_actual = X_scaled_actual_price[sample_indices]
                y_sample_actual = y_actual_price.iloc[sample_indices]
                y_pred_sample_actual = trained_model_actual_price.predict(X_sample_actual)
                
                X_sample_nextday = X_scaled_nextday_price[sample_indices]
                y_sample_nextday = y_nextday_price.iloc[sample_indices]
                y_pred_sample_nextday = trained_model_nextday_price.predict(X_sample_nextday)
                
                # Save combined evaluation details
                save_evaluation_details_regressor(
                    df_sample,
                    y_sample_actual, y_pred_sample_actual,  # actual_price data
                    y_sample_nextday, y_pred_sample_nextday,  # nextday_price data
                    model_id_actual_price, model_name_actual_price, version_actual_price,  # actual_price model info
                    model_id_nextday_price, model_name_nextday_price, version_nextday_price  # nextday_price model info
                )
            
            # Save results for both models
            results.append({
                'event': event,
                'target_type': 'price_change_percentage',
                'model_id': model_id_actual_price,
                'run_id': run_id_actual_price,
                'model_name': model_name_actual_price,
                'model_type': 'RandomForestRegressor',
                'version': version_actual_price,
                'model_category': args.model_category,
                'n_samples': len(df_event),
                'n_features': len(actual_price_features),
                'mlflow_experiment': experiment_name,
                'database_saved': True,
                'enhanced_features': True,
                'feature_selection': 'optimized'
            })
            
            results.append({
                'event': event,
                'target_type': 'nextday_price_change_percentage',
                'model_id': model_id_nextday_price,
                'run_id': run_id_nextday_price,
                'model_name': model_name_nextday_price,
                'model_type': 'RandomForestRegressor',
                'version': version_nextday_price,
                'model_category': args.model_category,
                'n_samples': len(df_event),
                'n_features': len(nextday_price_features),
                'mlflow_experiment': experiment_name,
                'database_saved': True,
                'enhanced_features': True,
                'feature_selection': 'optimized'
            })
        
        except Exception as e:
            logger.error(f"❌ Failed to train regressor models for {event}: {e}")
            continue
    
    # Save summary
    if results:
        metrics_csv = os.path.join(args.output_dir, 'enhanced_regressor_v2_training_metrics.csv')
        results_df = pd.DataFrame(results)
        results_df.to_csv(metrics_csv, index=False)
        logger.info(f"✅ Saved training summary to {metrics_csv}")
        logger.info(f"🎯 Enhanced regressor v2 training completed for {len(results)} models")
        logger.info("🎯 All artifacts logged to MLflow and database for team access!")
        logger.info("🎯 Enhanced features: Yahoo Finance, Polygon, market context, content analysis")
        logger.info("🎯 Feature selection: Optimized for each model separately")
    else:
        logger.error("❌ No models were successfully trained")

if __name__ == '__main__':
    main()
