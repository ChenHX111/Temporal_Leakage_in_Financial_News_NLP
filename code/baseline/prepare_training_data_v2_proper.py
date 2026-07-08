#!/usr/bin/env python3
"""
Prepare Training Data v2 - Proper Workflow
Based on prepare_training_data.py but with enhanced features and feature selection
"""

import os
import sys
import pandas as pd
import argparse
from dotenv import load_dotenv
import time
from datetime import datetime

# Load environment variables
load_dotenv()

# Import feature engineering and selection
from util.feature_engineering_v2 import add_enhanced_features_v2_ultra_fast
from util.feature_selection_v2 import clean_and_select_features
from util.save_training_data_to_db import save_v2_data_with_separate_columns

def setup_logger():
    import logging
    logger = logging.getLogger('Prepare_Training_Data_v2_Proper')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)
    return logger

def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None

def compute_side_from_change(change_series: pd.Series) -> pd.Series:
    def label(x):
        try:
            if pd.isna(x):
                return None
            if x > 0:
                return "up"
            elif x < 0:
                return "down"
            else:  # x == 0.0
                return "neutral"
        except Exception:
            return None
    return change_series.apply(label)

def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for c in columns:
        if c not in df.columns:
            df[c] = None
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the data - handle NaN values and data types"""
    logger = setup_logger()
    logger.info("🧹 Cleaning data...")
    
    df_clean = df.copy()
    original_shape = df_clean.shape
    
    # Handle missing values in numeric columns
    numeric_columns = df_clean.select_dtypes(include=['number']).columns
    for col in numeric_columns:
        if df_clean[col].isnull().any():
            # Fill with median for numeric columns
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            logger.info(f"   Filled {col} with median: {median_val}")
    
    # Handle missing values in text columns
    text_columns = df_clean.select_dtypes(include=['object']).columns
    for col in text_columns:
        if df_clean[col].isnull().any():
            # Fill with empty string for text columns
            df_clean[col] = df_clean[col].fillna('')
            logger.info(f"   Filled {col} with empty string")
    
    # Remove rows where target columns are completely missing
    target_columns = ['actual_side', 'nextday_side', 'price_change_percentage', 'nextday_price_change_percentage']
    for target in target_columns:
        if target in df_clean.columns:
            before_count = len(df_clean)
            df_clean = df_clean.dropna(subset=[target])
            after_count = len(df_clean)
            if before_count != after_count:
                logger.info(f"   Removed {before_count - after_count} rows with missing {target}")
    
    logger.info(f"✅ Data cleaning completed: {original_shape} -> {df_clean.shape}")
    return df_clean

def main():
    parser = argparse.ArgumentParser(description='Prepare Training Data v2 with Enhanced Features')
    parser.add_argument('--input-file', type=str, default=None, help='Input CSV file (default: auto-detect)')
    parser.add_argument('--n-features', type=int, default=30, help='Number of features to select (default: 30)')
    parser.add_argument('--add-features', action='store_true', default=True, help='Add enhanced features (default: True)')
    parser.add_argument('--select-features', action='store_true', default=True, help='Select best features (default: True)')
    parser.add_argument('--output-suffix', type=str, default='v2', help='Output file suffix (default: v2)')
    
    args = parser.parse_args()
    logger = setup_logger()
    start_time = time.time()
    
    logger.info("🚀 Starting V2 Training Data Preparation")
    logger.info(f"   Features to select: {args.n_features}")
    logger.info(f"   Add enhanced features: {args.add_features}")
    logger.info(f"   Select best features: {args.select_features}")
    
    # Step 1: Load and prepare base data (same as v1)
    candidates = [
        'data/price_moves_norm_cleaned.csv',
        'data/price_moves_norm_enriched.csv',
        'data/price_moves_norm_data.csv',
    ]
    input_file = args.input_file or next((p for p in candidates if os.path.exists(p)), None)
    if not input_file:
        raise FileNotFoundError("No input data found. Expected one of: " + ", ".join(candidates))

    logger.info(f"📊 Loading dataset from {input_file}")
    df = pd.read_csv(input_file)
    logger.info(f"   Loaded {len(df)} rows and {len(df.columns)} columns")
    
    # Step 2: Clean data
    df = clean_data(df)
    
    # Step 3: Prepare base data (same as v1)
    logger.info("📋 Preparing base data (same as v1)...")
    
    # Ensure mandatory feature columns for both outputs
    required_feature_cols = [
        'yf_ticker', 'exchange', 'etf_ticker', 'market_status',
        'title_en', 'content_en', 'event', 'publisher', 'published_date',
        'industry', 'publisher_topic'
    ]
    df = ensure_columns(df, required_feature_cols)
    
    # Add news_id column if it doesn't exist (for database linking)
    if 'news_id' not in df.columns:
        if 'id' in df.columns:
            df['news_id'] = df['id']
        else:
            df['news_id'] = range(1, len(df) + 1)
        logger.info(f"   Added news_id column for database linking")

    # Ensure ticker-based consolidation already done; fill yf_ticker from ticker if still missing
    if 'yf_ticker' in df.columns and 'ticker' in df.columns:
        df['yf_ticker'] = df['yf_ticker'].fillna(df['ticker'])

    # Make sure price change columns exist in a standard way if present under variants
    price_change_pct_col = first_existing_column(df, [
        'price_change_percentage', 'price_change_pct', 'pct_change'
    ])
    if price_change_pct_col and price_change_pct_col != 'price_change_percentage':
        df.rename(columns={price_change_pct_col: 'price_change_percentage'}, inplace=True)

    index_price_change_col = first_existing_column(df, [
        'index_price_change', 'idx_price_change', 'index_change'
    ])
    if index_price_change_col and index_price_change_col != 'index_price_change':
        df.rename(columns={index_price_change_col: 'index_price_change'}, inplace=True)

    index_price_change_pct_col = first_existing_column(df, [
        'index_price_change_percentage', 'idx_price_change_percentage', 'index_pct_change'
    ])
    if index_price_change_pct_col and index_price_change_pct_col != 'index_price_change_percentage':
        df.rename(columns={index_price_change_pct_col: 'index_price_change_percentage'}, inplace=True)

    # Next day price change percentage candidates
    nextday_col = first_existing_column(df, [
        'nextday_price_change_percentage', 'next_day_price_change_percentage',
        'nextday_price_change_pct', 'next_day_price_change_pct'
    ])
    if nextday_col and nextday_col != 'nextday_price_change_percentage':
        df.rename(columns={nextday_col: 'nextday_price_change_percentage'}, inplace=True)

    # Compute actual_side if missing
    if 'actual_side' not in df.columns and 'price_change_percentage' in df.columns:
        df['actual_side'] = compute_side_from_change(df['price_change_percentage'])

    # Compute nextday_side if nextday percentage exists
    if 'nextday_price_change_percentage' in df.columns:
        df['nextday_side'] = compute_side_from_change(df['nextday_price_change_percentage'])
    
    logger.info(f"✅ Base data prepared: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Step 4: Add enhanced features (if requested)
    if args.add_features:
        logger.info("🚀 Adding enhanced features v2...")
        try:
            df_enhanced = add_enhanced_features_v2_ultra_fast(df)
            logger.info(f"✅ Enhanced features added: {df_enhanced.shape[1] - df.shape[1]} new features")
            df = df_enhanced
        except Exception as e:
            logger.error(f"❌ Error adding enhanced features: {e}")
            logger.info("Continuing with base data...")
    
    # Step 5: Select best features (if requested)
    if args.select_features:
        logger.info(f"🎯 Selecting best {args.n_features} features...")
        try:
            target_columns = ['actual_side', 'nextday_side', 'price_change_percentage', 'nextday_price_change_percentage']
            df_selected = clean_and_select_features(df, target_columns, n_features=args.n_features)
            logger.info(f"✅ Feature selection completed: {df.shape[1]} -> {df_selected.shape[1]} columns")
            df = df_selected
        except Exception as e:
            logger.error(f"❌ Error in feature selection: {e}")
            logger.info("Continuing with all features...")
    
    # Step 6: Create output files
    logger.info("📁 Creating output files...")
    out_dir = os.path.join('data', 'training_data')
    os.makedirs(out_dir, exist_ok=True)
    
    # Build classifier training CSV
    classifier_extra_cols = [
        'price_change_percentage', 'price_change',
        'index_price_change', 'index_price_change_percentage',
        'actual_side', 'nextday_side', 'nextday_price_change_percentage'
    ]
    
    # Get all enhanced features that are not in the base required features
    enhanced_features = [col for col in df.columns if col not in required_feature_cols + classifier_extra_cols + ['news_id']]
    
    # Include news_id for database linking (but exclude from features)
    classifier_cols = ['news_id'] + [c for c in required_feature_cols + classifier_extra_cols if c in df.columns] + enhanced_features
    classifier_df = df[classifier_cols].copy()
    
    # Use timestamp to avoid permission conflicts
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    classifier_path = os.path.join(out_dir, f'classifier_training_{args.output_suffix}_{timestamp}.csv')
    classifier_df.to_csv(classifier_path, index=False)
    logger.info(f"✅ Classifier training data saved to {classifier_path} ({len(classifier_df)} rows, {len(classifier_df.columns)} cols)")

    # Save classifier training data to database with enhanced features as separate columns
    logger.info("Saving classifier training data to database (classifier_training_v2)...")
    if save_v2_data_with_separate_columns(classifier_df, "classifier_training_v2", enhanced_features):
        logger.info("✅ Classifier training data saved to database successfully")
    else:
        logger.error("❌ Failed to save classifier training data to database")

    # Build regression training CSV (exclude side labels)
    regression_extra_cols = [
        'price_change_percentage', 'price_change',
        'index_price_change', 'index_price_change_percentage',
        'nextday_price_change_percentage'
    ]
    
    # Include news_id for database linking (but exclude from features)
    regression_cols = ['news_id'] + [c for c in required_feature_cols + regression_extra_cols if c in df.columns] + enhanced_features
    regression_df = df[regression_cols].copy()
    regression_path = os.path.join(out_dir, f'regression_training_{args.output_suffix}_{timestamp}.csv')
    regression_df.to_csv(regression_path, index=False)
    logger.info(f"✅ Regression training data saved to {regression_path} ({len(regression_df)} rows, {len(regression_df.columns)} cols)")

    # Save regression training data to database with enhanced features as separate columns
    logger.info("Saving regression training data to database (regressor_training_v2)...")
    if save_v2_data_with_separate_columns(regression_df, "regressor_training_v2", enhanced_features):
        logger.info("✅ Regression training data saved to database successfully")
    else:
        logger.error("❌ Failed to save regression training data to database")
    
    # Also save the original files for backward compatibility
    classifier_original_path = os.path.join(out_dir, 'classifier_training.csv')
    classifier_original_df = df[['news_id'] + [c for c in required_feature_cols + classifier_extra_cols if c in df.columns]].copy()
    classifier_original_df.to_csv(classifier_original_path, index=False)
    logger.info(f"Original classifier training data saved to {classifier_original_path} ({len(classifier_original_df)} rows, {len(classifier_original_df.columns)} cols)")

    regression_original_path = os.path.join(out_dir, 'regression_training.csv')
    regression_original_df = df[['news_id'] + [c for c in required_feature_cols + regression_extra_cols if c in df.columns]].copy()
    regression_original_df.to_csv(regression_original_path, index=False)
    logger.info(f"Original regression training data saved to {regression_original_path} ({len(regression_original_df)} rows, {len(regression_original_df.columns)} cols)")
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    logger.info("🎯 V2 Training Data Preparation Completed!")
    logger.info(f"📊 Enhanced features added: {len(enhanced_features)}")
    logger.info(f"⏱️ Total processing time: {processing_time:.2f} seconds ({processing_time/60:.2f} minutes)")
    logger.info(f"📈 Processing speed: {len(df)/processing_time:.0f} records/second")
    logger.info("📁 Files created:")
    logger.info(f"   - {classifier_path} (enhanced classifier data)")
    logger.info(f"   - {regression_path} (enhanced regression data)")
    logger.info(f"   - {classifier_original_path} (original classifier data)")
    logger.info(f"   - {regression_original_path} (original regression data)")
    
    # Show sample of features
    if enhanced_features:
        logger.info("🔍 Sample of enhanced features:")
        feature_sample = enhanced_features[:10]  # Show first 10 features
        for feature in feature_sample:
            logger.info(f"   - {feature}")
        if len(enhanced_features) > 10:
            logger.info(f"   ... and {len(enhanced_features) - 10} more features")

if __name__ == '__main__':
    main()
