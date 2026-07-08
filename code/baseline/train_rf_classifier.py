#!/usr/bin/env python3
"""
Enhanced Classifier Training Script with Database Artifact Storage
Stores ALL artifacts (scripts, datasets, models, configs) in database BLOB storage
Uses RandomForestClassifier for price direction prediction
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
from util.mlflow_setup import setup_logger, setup_mlflow, train_and_save_classifier
from util.save_to_db import save_classifier_to_database, save_evaluation_details_classifier






def main():
    logger = setup_logger()
    logger.info("🚀 Starting Enhanced Random Forest Classifier Training...")
    logger.info("🎯 Comprehensive MLflow logging with database artifact storage for team collaboration")

    parser = argparse.ArgumentParser(description='Enhanced RF Classifier Training')
    parser.add_argument('--input-file', type=str, default='data/training_data/classifier_training.csv')
    parser.add_argument('--output-dir', type=str, default='reports/enhanced_classifier_training')
    parser.add_argument('--model-category', type=str, default='ml_models', choices=['ml_models', 'rag_models', 'llm_models'])
    parser.add_argument('--event', type=str, default='all_events', help='Specific event to train on')
    args = parser.parse_args()

    # Setup MLflow
    mlflow_available, experiment_name, model_category = setup_mlflow('classifier', args.model_category)
    if not mlflow_available:
        logger.error("❌ MLflow setup failed. Cannot continue without MLflow.")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Loading dataset from {args.input_file}")
    df = pd.read_csv(args.input_file)
    logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")

    # Check for required target columns
    required_targets = ['actual_side']
    missing_targets = [target for target in required_targets if target not in df.columns]
    if missing_targets:
        raise ValueError(f"Missing required target columns: {missing_targets}")

    # Filter to up/down only for model training
    df = df[df['actual_side'].isin(['up', 'down'])].copy()
    if df.empty:
        raise ValueError("No samples with actual_side in ['up','down'] after filtering")

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
    feature_cols_actual_side = base_features + ['event']
    feature_cols_nextday_side = base_features + ['event', 'actual_side']
    
    # Add TF-IDF feature names
    tfidf_feature_names = []
    tfidf_feature_names.extend([f"title_tfidf_{i}" for i in range(title_features_dense.shape[1])])
    tfidf_feature_names.extend([f"content_tfidf_{i}" for i in range(content_features_dense.shape[1])])
    
    logger.info(f"Total features for actual_side: {len(feature_cols_actual_side) + len(tfidf_feature_names)}")
    logger.info(f"Total features for nextday_side: {len(feature_cols_nextday_side) + len(tfidf_feature_names)}")
    
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
        
        # Train dual classifier models
        logger.info(f"🎯 Training dual classifier models for {event}")
        
        # Prepare data
        df_processed = df_event.copy()
        for col in base_features:
            if df_processed[col].dtype == 'object':
                df_processed[col] = pd.Categorical(df_processed[col].astype(str)).codes
        
        if df_processed['event'].dtype == 'object':
            df_processed['event'] = pd.Categorical(df_processed['event'].astype(str)).codes
        
        # Handle actual_side and nextday_side features (categorical encoding)
        if df_processed['actual_side'].dtype == 'object':
            df_processed['actual_side'] = pd.Categorical(df_processed['actual_side'].astype(str)).codes
        if df_processed['nextday_side'].dtype == 'object':
            df_processed['nextday_side'] = pd.Categorical(df_processed['nextday_side'].astype(str)).codes
        
        df_processed = df_processed.fillna(0)
        df_processed[base_features] = df_processed[base_features].astype(float)
        df_processed['event'] = df_processed['event'].astype(float)
        df_processed['actual_side'] = df_processed['actual_side'].astype(float)
        df_processed['nextday_side'] = df_processed['nextday_side'].astype(float)
        
        # Get TF-IDF features
        title_features_event = title_tfidf.transform(df_event['title_en'].fillna(''))
        content_features_event = content_tfidf.transform(df_event['content_en'].fillna(''))
        
        title_features_event_dense = title_features_event.toarray()
        content_features_event_dense = content_features_event.toarray()
                
        # Train Model 1: actual_side
        try:
            logger.info(f"🎯 Training Model 1: actual_side classifier")
            y_actual_side = df_event['actual_side'].str.lower()
            X_actual_side = df_processed[feature_cols_actual_side].copy()
            
            X_actual_side_combined = np.hstack([X_actual_side.values, title_features_event_dense, content_features_event_dense])
            
            scaler_actual_side = StandardScaler()
            X_scaled_actual_side = scaler_actual_side.fit_transform(X_actual_side_combined)
            
            model_id_actual_side, run_id_actual_side, trained_model_actual_side, version_actual_side, model_name_actual_side, le_actual_side, training_summary_actual_side = train_and_save_classifier(
                X_scaled_actual_side, y_actual_side, 
                event_name=event,
                target_type='actual_side',
                model_category=args.model_category,
                experiment_name=experiment_name,
                feature_columns=feature_cols_actual_side + tfidf_feature_names,
                scaler=scaler_actual_side,
                title_tfidf=title_tfidf,
                content_tfidf=content_tfidf,
                dataset_blob=dataset_blob,
                script_blob=script_blob
            )
            
            # Save model to database with BLOB objects
            save_classifier_to_database(
                model_id=model_id_actual_side,
                event_type=event,
                target_type='actual_side',
                version=version_actual_side,
                accuracy=training_summary_actual_side.get('accuracy'),
                f1_score=training_summary_actual_side.get('f1_score'),
                precision=training_summary_actual_side.get('precision'),
                recall=training_summary_actual_side.get('recall'),
                roc_auc=training_summary_actual_side.get('roc_auc'),
                train_samples=training_summary_actual_side.get('train_samples'),
                test_samples=training_summary_actual_side.get('test_samples'),
                cv_folds=training_summary_actual_side.get('cv_folds'),
                best_params={"n_estimators": 200, "max_depth": 20, "method": "standard_training", "target_type": 'actual_side'},
                feature_columns=feature_cols_actual_side + tfidf_feature_names,
                mlflow_run_id=run_id_actual_side,
                experiment_name=experiment_name,
                model_name=model_name_actual_side,
                model_stage="Staging",
                model_category=args.model_category,
                model_obj=trained_model_actual_side,
                scaler_obj=scaler_actual_side,
                encoder_obj=le_actual_side,
                title_tfidf_obj=title_tfidf,
                content_tfidf_obj=content_tfidf
            )
            
            logger.info(f"✅ Successfully trained actual_side classifier for {event}")
            
        except Exception as e:
            logger.error(f"❌ Failed to train actual_side classifier for {event}: {e}")
            continue
        
        # Train Model 2: nextday_side (using actual_side as feature)
        try:
            logger.info(f"🎯 Training Model 2: nextday_side classifier")
            y_nextday_side = df_event['nextday_side'].str.lower()
            X_nextday_side = df_processed[feature_cols_nextday_side].copy()
            
            X_nextday_side_combined = np.hstack([X_nextday_side.values, title_features_event_dense, content_features_event_dense])
            
            scaler_nextday_side = StandardScaler()
            X_scaled_nextday_side = scaler_nextday_side.fit_transform(X_nextday_side_combined)
            
            model_id_nextday_side, run_id_nextday_side, trained_model_nextday_side, version_nextday_side, model_name_nextday_side, le_nextday_side, training_summary_nextday_side = train_and_save_classifier(
                X_scaled_nextday_side, y_nextday_side, 
                event_name=event,
                target_type='nextday_side',
                model_category=args.model_category,
                experiment_name=experiment_name,
                feature_columns=feature_cols_nextday_side + tfidf_feature_names,
                scaler=scaler_nextday_side,
                title_tfidf=title_tfidf,
                content_tfidf=content_tfidf,
                dataset_blob=dataset_blob,
                script_blob=script_blob
            )
            
            # Save model to database with BLOB objects
            save_classifier_to_database(
                model_id=model_id_nextday_side,
                event_type=event,
                target_type='nextday_side',
                version=version_nextday_side,
                accuracy=training_summary_nextday_side.get('accuracy'),
                f1_score=training_summary_nextday_side.get('f1_score'),
                precision=training_summary_nextday_side.get('precision'),
                recall=training_summary_nextday_side.get('recall'),
                roc_auc=training_summary_nextday_side.get('roc_auc'),
                train_samples=training_summary_nextday_side.get('train_samples'),
                test_samples=training_summary_nextday_side.get('test_samples'),
                cv_folds=training_summary_nextday_side.get('cv_folds'),
                best_params={"n_estimators": 200, "max_depth": 20, "method": "standard_training", "target_type": 'nextday_side'},
                feature_columns=feature_cols_nextday_side + tfidf_feature_names,
                mlflow_run_id=run_id_nextday_side,
                experiment_name=experiment_name,
                model_name=model_name_nextday_side,
                model_stage="Staging",
                model_category=args.model_category,
                model_obj=trained_model_nextday_side,
                scaler_obj=scaler_nextday_side,
                encoder_obj=le_nextday_side,
                title_tfidf_obj=title_tfidf,
                content_tfidf_obj=content_tfidf
            )
            
            logger.info(f"✅ Successfully trained nextday_side classifier for {event}")
            
        except Exception as e:
            logger.error(f"❌ Failed to train nextday_side classifier for {event}: {e}")
            continue
        
        # Save evaluation details for both models
        if len(df_event) > 800:
            sample_size = min(100, len(df_event))
            sample_indices = np.random.choice(len(df_event), sample_size, replace=False)
            df_sample = df_event.iloc[sample_indices].reset_index(drop=True)
            
            # Evaluation for both models combined
            X_sample_actual = X_scaled_actual_side[sample_indices]
            y_sample_actual = y_actual_side.iloc[sample_indices]
            y_pred_sample_actual = trained_model_actual_side.predict(X_sample_actual)
            
            X_sample_nextday = X_scaled_nextday_side[sample_indices]
            y_sample_nextday = y_nextday_side.iloc[sample_indices]
            y_pred_sample_nextday = trained_model_nextday_side.predict(X_sample_nextday)
            
            # Save combined evaluation details
            save_evaluation_details_classifier(
                df_sample, 
                y_sample_actual, y_pred_sample_actual,  # actual_side data
                y_sample_nextday, y_pred_sample_nextday,  # nextday_side data
                model_id_actual_side, model_name_actual_side, version_actual_side,  # actual_side model info
                model_id_nextday_side, model_name_nextday_side, version_nextday_side  # nextday_side model info
            )
        
        # Save results for both models
        results.append({
            'event': event,
            'target_type': 'actual_side',
            'model_id': model_id_actual_side,
            'run_id': run_id_actual_side,
            'model_name': model_name_actual_side,
            'model_type': 'RandomForestClassifier',
            'version': version_actual_side,
            'model_category': args.model_category,
            'n_samples': len(df_event),
            'n_features': len(feature_cols_actual_side),
            'mlflow_experiment': experiment_name,
            'database_saved': True
        })
        
        results.append({
            'event': event,
            'target_type': 'nextday_side',
            'model_id': model_id_nextday_side,
            'run_id': run_id_nextday_side,
            'model_name': model_name_nextday_side,
            'model_type': 'RandomForestClassifier',
            'version': version_nextday_side,
            'model_category': args.model_category,
            'n_samples': len(df_event),
            'n_features': len(feature_cols_nextday_side),
            'mlflow_experiment': experiment_name,
            'database_saved': True
        })
    
    # Save summary
    if results:
        metrics_csv = os.path.join(args.output_dir, 'enhanced_classifier_training_metrics.csv')
        results_df = pd.DataFrame(results)
        results_df.to_csv(metrics_csv, index=False)
        logger.info(f"✅ Saved training summary to {metrics_csv}")
        logger.info(f"🎯 Enhanced classifier training completed for {len(results)} models")
        logger.info("🎯 All artifacts logged to MLflow and database for team access!")
    else:
        logger.error("❌ No models were successfully trained")

if __name__ == '__main__':
    main()
