#!/usr/bin/env python3
"""
Enhanced Regressor Training Script with Database Artifact Storage
Stores ALL artifacts (scripts, datasets, models, configs) in database BLOB storage
Uses RandomForestRegressor for price change prediction
Team members can access everything from MLflow UI without file system access
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Import modular functions
from util.mlflow_setup import setup_logger, setup_mlflow, train_and_save_regressor
from util.save_to_db import save_regressor_to_database, save_evaluation_details_regressor






def main():
    logger = setup_logger()
    logger.info("🚀 Starting Enhanced Random Forest Regressor Training...")
    logger.info("🎯 Comprehensive MLflow logging with database artifact storage for team collaboration")

    parser = argparse.ArgumentParser(description='Enhanced RF Regressor Training')
    parser.add_argument('--input-file', type=str, default='data/training_data/regression_training.csv')
    parser.add_argument('--output-dir', type=str, default='reports/enhanced_regressor_training')
    parser.add_argument('--model-category', type=str, default='ml_models', choices=['ml_models', 'rag_models', 'llm_models'])
    parser.add_argument('--event', type=str, default='clinical_study', help='Specific event to train on')
    args = parser.parse_args()

    # Setup MLflow
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

    # Define base features
    base_features = [
        'yf_ticker', 'exchange', 'etf_ticker', 'market_status',
        'publisher', 'published_date', 'industry', 'publisher_topic'
    ]
    
    # Create TF-IDF features
    logger.info("Creating TF-IDF features for text content...")
    
    title_tfidf = TfidfVectorizer(
        max_features=100,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words='english'
    )
    
    content_tfidf = TfidfVectorizer(
        max_features=200,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words='english'
    )
    
    # Fit and transform text features
    title_features = title_tfidf.fit_transform(df['title_en'].fillna(''))
    content_features = content_tfidf.fit_transform(df['content_en'].fillna(''))
    
    title_features_dense = title_features.toarray()
    content_features_dense = content_features.toarray()
    
    logger.info(f"Title TF-IDF features: {title_features_dense.shape[1]}")
    logger.info(f"Content TF-IDF features: {content_features_dense.shape[1]}")
    
    # Feature sets for dual models
    feature_cols_actual_price = base_features + ['event']
    feature_cols_nextday_price = base_features + ['event', 'price_change_percentage', 'index_price_change_percentage', 'price_change']
    
    # Add TF-IDF feature names
    tfidf_feature_names = []
    tfidf_feature_names.extend([f"title_tfidf_{i}" for i in range(title_features_dense.shape[1])])
    tfidf_feature_names.extend([f"content_tfidf_{i}" for i in range(content_features_dense.shape[1])])
    
    logger.info(f"Total features for actual_price_change: {len(feature_cols_actual_price) + len(tfidf_feature_names)}")
    logger.info(f"Total features for nextday_price_change: {len(feature_cols_nextday_price) + len(tfidf_feature_names)}")
    
    # Train for specific event or all events
    if args.event == 'all_events':
        # Train ONLY on combined dataset (all events together)
        target_events = ['all_events']
        logger.info(f"Training ONLY on combined dataset (all events together)")
        logger.info(f"Target events: {target_events}")
    elif args.event == 'all':
        # Get all events with sufficient samples
        event_counts = df['event'].value_counts()
        individual_events = event_counts[event_counts >= 500].index.tolist()
        target_events = ['all_events'] + individual_events
        logger.info(f"Training on combined dataset + {len(individual_events)} individual events with ≥800 samples")
        logger.info(f"Target events: {target_events}")
    else:
        target_events = [args.event]
        logger.info(f"Training on specific event: {args.event}")
    
    results = []
    
    for event in target_events:
        if event == 'all_events':
            df_event = df.copy()
        else:
            df_event = df[df['event'] == event].copy()
            
        if len(df_event) < 500:
            logger.info(f"Skipping event {event} (only {len(df_event)} samples, need ≥800)")
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
            for col in base_features:
                if df_processed[col].dtype == 'object':
                    df_processed[col] = pd.Categorical(df_processed[col].astype(str)).codes
            
            if df_processed['event'].dtype == 'object':
                df_processed['event'] = pd.Categorical(df_processed['event'].astype(str)).codes
            
            # Handle price change features (already numeric)
            df_processed['price_change_percentage'] = df_processed['price_change_percentage'].fillna(0)
            df_processed['nextday_price_change_percentage'] = df_processed['nextday_price_change_percentage'].fillna(0)
            df_processed['index_price_change_percentage'] = df_processed['index_price_change_percentage'].fillna(0)
            df_processed['price_change'] = df_processed['price_change'].fillna(0)
            
            df_processed = df_processed.fillna(0)
            df_processed[base_features] = df_processed[base_features].astype(float)
            df_processed['event'] = df_processed['event'].astype(float)
            df_processed['price_change_percentage'] = df_processed['price_change_percentage'].astype(float)
            df_processed['nextday_price_change_percentage'] = df_processed['nextday_price_change_percentage'].astype(float)
            df_processed['index_price_change_percentage'] = df_processed['index_price_change_percentage'].astype(float)
            df_processed['price_change'] = df_processed['price_change'].astype(float)
            
            # Get TF-IDF features
            title_features_event = title_tfidf.transform(df_event['title_en'].fillna(''))
            content_features_event = content_tfidf.transform(df_event['content_en'].fillna(''))
            
            title_features_event_dense = title_features_event.toarray()
            content_features_event_dense = content_features_event.toarray()
            
            # Train Model 1: price_change_percentage
            try:
                logger.info(f"🎯 Training Model 1: price_change_percentage regressor")
                y_actual_price = df_event['price_change_percentage']
                X_actual_price = df_processed[feature_cols_actual_price].copy()
                
                X_actual_price_combined = np.hstack([X_actual_price.values, title_features_event_dense, content_features_event_dense])
                
                scaler_actual_price = StandardScaler()
                X_scaled_actual_price = scaler_actual_price.fit_transform(X_actual_price_combined)
                
                model_id_actual_price, run_id_actual_price, trained_model_actual_price, version_actual_price, model_name_actual_price, training_summary_actual_price = train_and_save_regressor(
                    X_scaled_actual_price, y_actual_price, 
                    event_name=event,
                    target_type='price_change_percentage',
                    model_category=args.model_category,
                    experiment_name=experiment_name,
                    feature_columns=feature_cols_actual_price + tfidf_feature_names,
                    scaler=scaler_actual_price,
                    title_tfidf=title_tfidf,
                    content_tfidf=content_tfidf,
                    dataset_blob=dataset_blob,
                    script_blob=script_blob
                )
                
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
                    best_params={"n_estimators": 200, "max_depth": 20, "method": "standard_training", "target_type": 'price_change_percentage'},
                    feature_columns=feature_cols_actual_price + tfidf_feature_names,
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
                X_nextday_price = df_processed[feature_cols_nextday_price].copy()
                
                X_nextday_price_combined = np.hstack([X_nextday_price.values, title_features_event_dense, content_features_event_dense])
                
                scaler_nextday_price = StandardScaler()
                X_scaled_nextday_price = scaler_nextday_price.fit_transform(X_nextday_price_combined)
                
                model_id_nextday_price, run_id_nextday_price, trained_model_nextday_price, version_nextday_price, model_name_nextday_price, training_summary_nextday_price = train_and_save_regressor(
                    X_scaled_nextday_price, y_nextday_price, 
                    event_name=event,
                    target_type='nextday_price_change_percentage',
                    model_category=args.model_category,
                    experiment_name=experiment_name,
                    feature_columns=feature_cols_nextday_price + tfidf_feature_names,
                    scaler=scaler_nextday_price,
                    title_tfidf=title_tfidf,
                    content_tfidf=content_tfidf,
                    dataset_blob=dataset_blob,
                    script_blob=script_blob
                )
                
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
                    best_params={"n_estimators": 200, "max_depth": 20, "method": "standard_training", "target_type": 'nextday_price_change_percentage'},
                    feature_columns=feature_cols_nextday_price + tfidf_feature_names,
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
                'n_features': len(feature_cols_actual_price),
                'mlflow_experiment': experiment_name,
                'database_saved': True
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
                'n_features': len(feature_cols_nextday_price),
                'mlflow_experiment': experiment_name,
                'database_saved': True
            })
        
        except Exception as e:
            logger.error(f"❌ Failed to train regressor models for {event}: {e}")
            continue
    
    # Save summary
    if results:
        metrics_csv = os.path.join(args.output_dir, 'enhanced_regressor_training_metrics.csv')
        results_df = pd.DataFrame(results)
        results_df.to_csv(metrics_csv, index=False)
        logger.info(f"✅ Saved training summary to {metrics_csv}")
        logger.info(f"🎯 Enhanced regressor training completed for {len(results)} models")
        logger.info("🎯 All artifacts logged to MLflow and database for team access!")
    else:
        logger.error("❌ No models were successfully trained")

if __name__ == '__main__':
    main()
