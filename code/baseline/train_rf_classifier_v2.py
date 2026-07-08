#!/usr/bin/env python3
"""
Enhanced Classifier Training Script v2 with Advanced Feature Engineering
Uses enhanced features from Yahoo Finance, Polygon, market context, and content analysis
Optimizes features for each model separately for best accuracy
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Import modular functions
from util.mlflow_setup import setup_logger, setup_mlflow, train_and_save_classifier
from util.save_to_db import save_classifier_to_database, save_evaluation_details_classifier
from util.feature_engineering_v2 import add_enhanced_features_v2_ultra_fast
from util.feature_selection_v2 import clean_and_select_features

# Global wrapper class to avoid serialization issues
class AdvancedModelWrapper:
    def __init__(self, ensemble_model, poly_transformer, scaler):
        self.ensemble_model = ensemble_model
        self.poly_transformer = poly_transformer
        self.scaler = scaler
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        if self.poly_transformer is not None:
            X_poly = self.poly_transformer.transform(X_scaled)
            return self.ensemble_model.predict(X_poly)
        else:
            return self.ensemble_model.predict(X_scaled)
    
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        if self.poly_transformer is not None:
            X_poly = self.poly_transformer.transform(X_scaled)
            return self.ensemble_model.predict_proba(X_poly)
        else:
            return self.ensemble_model.predict_proba(X_scaled)

def train_simple_classifier(X, y, target_type, logger, n_features=30):
    """Train simple Random Forest with basic grid search (max 10 fits)"""
    logger.info(f"🚀 Training simple Random Forest for {target_type}")
    
    # Create simple Random Forest
    rf = RandomForestClassifier(
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
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=cv,
        scoring='f1_weighted',
        n_jobs=1,
        verbose=1
    )
    
    logger.info("🔍 Running simple grid search (max 10 fits)...")
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_
    best_cv_score = grid_search.best_score_
    
    logger.info(f"✅ Best CV score: {best_cv_score:.4f}")
    logger.info(f"✅ Best params: {grid_search.best_params_}")
    
    return best_model, None, best_cv_score

def main():
    logger = setup_logger()
    logger.info("🚀 Starting Enhanced Random Forest Classifier Training v2...")
    logger.info("🎯 Advanced feature engineering with Yahoo Finance, Polygon, market context, and content analysis")
    logger.info("🎯 Optimized feature selection for each model separately")

    parser = argparse.ArgumentParser(description='Enhanced RF Classifier Training v2')
    parser.add_argument('--input-file', type=str, default=r'D:\Oxford\Extra\Finance_NLP\finespresso-modelling\data\training_data\classifier_training_v2_202509180139.csv')
    parser.add_argument('--output-dir', type=str, default=r'D:\Oxford\Extra\Finance_NLP\finespresso-modelling\data\training_data\enhanced_classifier_v2_training')
    parser.add_argument('--model-category', type=str, default='ml_models', choices=['ml_models', 'rag_models', 'llm_models'])
    parser.add_argument('--event', type=str, default='partnerships', help='Specific event to train on (partnerships, all_events, or all)')
    parser.add_argument('--polygon-api-key', type=str, default=None, help='Polygon.io API key for enhanced features')
    parser.add_argument('--n-features', type=int, default=30, help='Number of features to select per model (default: 30)')
    args = parser.parse_args()

    # Setup MLflow
    mlflow_available, experiment_name, model_category = setup_mlflow('classifier_v2', args.model_category)
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

    # Data is already enhanced with v2 features, just clean and select
    logger.info("🎯 Cleaning and selecting best features...")
    target_columns = ['actual_side', 'nextday_side']
    df_selected = clean_and_select_features(df, target_columns, n_features=args.n_features)
    logger.info(f"✅ Selected dataset shape: {df_selected.shape}")
    
    # Ensure we have the required target columns for nextday prediction
    if 'price_change_percentage' not in df_selected.columns:
        logger.warning("⚠️ price_change_percentage not found in dataset, adding placeholder")
        df_selected['price_change_percentage'] = 0.0

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
        
        # Train dual classifier models
        logger.info(f"🎯 Training dual classifier models for {event}")
        
        # Prepare data
        df_processed = df_event.copy()
        
        # Get feature columns (exclude target columns and text columns)
        feature_columns = [col for col in df_processed.columns 
                          if col not in ['actual_side', 'nextday_side', 'news_id', 'title_en', 'content_en', 
                                       'title', 'content', 'ticker', 'company', 'reason', 'link', 'ticker_url', 
                                       'downloaded_at', 'timezone', 'publisher_summary', 'language', 'yf_ticker.1']]
        
        # Handle categorical encoding for feature columns
        for col in feature_columns:
            if col in df_processed.columns and df_processed[col].dtype == 'object':
                df_processed[col] = pd.Categorical(df_processed[col].astype(str)).codes
        
        # Special handling for actual_side when used as feature
        if 'actual_side' in df_processed.columns and df_processed['actual_side'].dtype == 'object':
            # Encode actual_side as numeric for use as feature
            actual_side_mapping = {'up': 1, 'down': 0, 'neutral': 0.5}
            df_processed['actual_side'] = df_processed['actual_side'].map(actual_side_mapping).fillna(0.5)
            logger.info("✅ Encoded actual_side as numeric feature: up=1, down=0, neutral=0.5")
        
        df_processed = df_processed.fillna(0)
        for col in feature_columns:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].astype(float)
        
        # Get TF-IDF features
        title_features_event = title_tfidf.transform(df_event['title_en'].fillna(''))
        content_features_event = content_tfidf.transform(df_event['content_en'].fillna(''))
        
        title_features_event_dense = title_features_event.toarray()
        content_features_event_dense = content_features_event.toarray()
        
        # Train Model 1: actual_side
        try:
            logger.info(f"🎯 Training Model 1: actual_side classifier")
            y_actual_side = df_event['actual_side'].str.lower()
            
            # Use features for actual_side (EXCLUDE all price change features - they are targets/leakage)
            actual_side_features = [col for col in feature_columns 
                                  if col not in ['price_change', 'index_price_change', 'index_price_change_percentage',
                                               'price_change_percentage', 'nextday_price_change_percentage',
                                               'nextday_price_change', 'nextday_index_change', 'nextday_index_change_percentage',
                                               'actual_side', 'nextday_side']]
            logger.info(f"Using {len(actual_side_features)} features for actual_side (excluding all price change features)")
            
            # Prepare features
            X_actual_side = df_processed[actual_side_features].copy()
            X_actual_side_combined = np.hstack([X_actual_side.values, title_features_event_dense, content_features_event_dense])
            
            scaler_actual_side = StandardScaler()
            X_scaled_actual_side = scaler_actual_side.fit_transform(X_actual_side_combined)
            
            # Add TF-IDF feature names
            tfidf_feature_names = []
            tfidf_feature_names.extend([f"title_tfidf_{i}" for i in range(title_features_event_dense.shape[1])])
            tfidf_feature_names.extend([f"content_tfidf_{i}" for i in range(content_features_event_dense.shape[1])])
            
            # Use simple Random Forest training
            logger.info("🎯 Using simple Random Forest training for actual_side...")
            trained_model_actual_side, poly_transformer_actual_side, cv_score_actual_side = train_simple_classifier(
                X_scaled_actual_side, y_actual_side, 'actual_side', logger, args.n_features
            )
            
            # Use the global wrapper class
            
            # Wrap the advanced model
            wrapped_model = AdvancedModelWrapper(trained_model_actual_side, poly_transformer_actual_side, scaler_actual_side)
            
            # Use basic MLflow training for logging (but with our advanced model)
            model_id_actual_side, run_id_actual_side, _, version_actual_side, model_name_actual_side, le_actual_side, training_summary_actual_side = train_and_save_classifier(
                X_scaled_actual_side, y_actual_side, 
                event_name=event,
                target_type='actual_side',
                model_category=args.model_category,
                experiment_name=experiment_name,
                feature_columns=actual_side_features + tfidf_feature_names,
                scaler=scaler_actual_side,
                title_tfidf=title_tfidf,
                content_tfidf=content_tfidf,
                dataset_blob=dataset_blob,
                script_blob=script_blob,
                method="enhanced_v2_training"
            )
            
            # Replace with our advanced model
            trained_model_actual_side = wrapped_model
            
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
                best_params={"n_estimators": 200, "max_depth": 20, "method": "enhanced_v2_training", "target_type": 'actual_side', "version": "v2"},
                feature_columns=actual_side_features + tfidf_feature_names,
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
        
        # Train Model 2: nextday_side (using actual_side and price_change_percentage as features)
        try:
            logger.info(f"🎯 Training Model 2: nextday_side classifier")
            y_nextday_side = df_event['nextday_side'].str.lower()
            
            # Use features for nextday_side (INCLUDE price change features and actual_side, EXCLUDE nextday leakage features)
            nextday_side_features = [col for col in feature_columns 
                                   if col not in ['nextday_price_change', 'nextday_index_change', 'nextday_price_change_percentage', 'nextday_side']]
            
            # Add actual_side as feature for nextday prediction (like v1)
            if 'actual_side' in df_processed.columns and 'actual_side' not in nextday_side_features:
                nextday_side_features.append('actual_side')
                logger.info("✅ Added actual_side as feature for nextday_side prediction")
            
            # Add price_change_percentage as feature for nextday prediction (like v1)
            if 'price_change_percentage' in df_processed.columns and 'price_change_percentage' not in nextday_side_features:
                nextday_side_features.append('price_change_percentage')
                logger.info("✅ Added price_change_percentage as feature for nextday_side prediction")
            
            logger.info(f"Using {len(nextday_side_features)} features for nextday_side (including actual_side and price_change_percentage)")
            
            # Prepare features
            X_nextday_side = df_processed[nextday_side_features].copy()
            X_nextday_side_combined = np.hstack([X_nextday_side.values, title_features_event_dense, content_features_event_dense])
            
            scaler_nextday_side = StandardScaler()
            X_scaled_nextday_side = scaler_nextday_side.fit_transform(X_nextday_side_combined)
            
            # Use simple Random Forest training for nextday_side
            logger.info("🎯 Using simple Random Forest training for nextday_side...")
            trained_model_nextday_side, poly_transformer_nextday_side, cv_score_nextday_side = train_simple_classifier(
                X_scaled_nextday_side, y_nextday_side, 'nextday_side', logger, args.n_features
            )
            
            # Wrap the advanced model for nextday_side
            wrapped_model_nextday_side = AdvancedModelWrapper(trained_model_nextday_side, poly_transformer_nextday_side, scaler_nextday_side)
            
            # Use basic MLflow training for logging (but with our advanced model)
            model_id_nextday_side, run_id_nextday_side, _, version_nextday_side, model_name_nextday_side, le_nextday_side, training_summary_nextday_side = train_and_save_classifier(
                X_scaled_nextday_side, y_nextday_side, 
                event_name=event,
                target_type='nextday_side',
                model_category=args.model_category,
                experiment_name=experiment_name,
                feature_columns=nextday_side_features + tfidf_feature_names,
                scaler=scaler_nextday_side,
                title_tfidf=title_tfidf,
                content_tfidf=content_tfidf,
                dataset_blob=dataset_blob,
                script_blob=script_blob,
                method="enhanced_v2_training"
            )
            
            # Replace with our advanced model
            trained_model_nextday_side = wrapped_model_nextday_side
            
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
                best_params={"n_estimators": 200, "max_depth": 20, "method": "enhanced_v2_training", "target_type": 'nextday_side', "version": "v2"},
                feature_columns=nextday_side_features + tfidf_feature_names,
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
            'n_features': len(actual_side_features),
            'mlflow_experiment': experiment_name,
            'database_saved': True,
            'enhanced_features': True,
            'feature_selection': 'optimized'
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
            'n_features': len(nextday_side_features),
            'mlflow_experiment': experiment_name,
            'database_saved': True,
            'enhanced_features': True,
            'feature_selection': 'optimized'
        })
    
    # Save summary
    if results:
        metrics_csv = os.path.join(args.output_dir, 'enhanced_classifier_v2_training_metrics.csv')
        results_df = pd.DataFrame(results)
        results_df.to_csv(metrics_csv, index=False)
        logger.info(f"✅ Saved training summary to {metrics_csv}")
        logger.info(f"🎯 Enhanced classifier v2 training completed for {len(results)} models")
        logger.info("🎯 All artifacts logged to MLflow and database for team access!")
        logger.info("🎯 Enhanced features: Yahoo Finance, Polygon, market context, content analysis")
        logger.info("🎯 Feature selection: Optimized for each model separately")
    else:
        logger.error("❌ No models were successfully trained")

if __name__ == '__main__':
    main()
