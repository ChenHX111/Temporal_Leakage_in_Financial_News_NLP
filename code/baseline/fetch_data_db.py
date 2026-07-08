import pandas as pd
import json
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Load environment variables from .env file
load_dotenv()

def setup_logger():
    import logging
    logger = logging.getLogger('ML_Improvement_Data_Extraction')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)
    return logger

def main():
    logger = setup_logger()
    
    # Get the database URL from .env
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL not found in .env file")
        raise ValueError("DATABASE_URL not found in .env file")

    logger.info("Creating database connection...")
    # Create a database connection
    engine = create_engine(database_url)

    # Define the SQL query to merge tables and select all price_moves_norm columns plus selected news columns
    query = """
    SELECT
        p.*,
        n.content, n.title, n.content_en, n.title_en,
        n.event, n.publisher, n.published_date, n.ticker, n.company, n.reason, n.link, n.ticker_url,
        n.industry, n.downloaded_at, n.timezone, n.publisher_summary, n.language, n.publisher_topic,
        n.yf_ticker
    FROM news n
    INNER JOIN "price_moves_norm" p
        ON n.id = p.news_id
    """

    logger.info("Executing query to merge news and price_moves tables...")
    # Execute the query and load into a DataFrame
    df = pd.read_sql(query, engine)
    
    logger.info(f"Retrieved {len(df)} records from database")

    # Ensure the 'data' directory exists
    os.makedirs("data", exist_ok=True)

    # Save the raw merged data first
    raw_output_path = "data/price_moves_norm_data.csv"
    df.to_csv(raw_output_path, index=False)
    
    logger.info(f"Raw merged data saved to '{raw_output_path}'")
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")

    # ---------------------------------------------
    # Enrichment steps
    # ---------------------------------------------
    # Consolidate potential duplicate ticker/yf_ticker columns and fill yf_ticker from ticker where missing
    def first_existing_column(columns):
        return next((c for c in columns if c in df.columns), None)

    # Resolve ticker column
    ticker_col = first_existing_column(["ticker", "ticker_x", "ticker_y"])
    if ticker_col and ticker_col != "ticker":
        df.rename(columns={ticker_col: "ticker"}, inplace=True)

    # Resolve yf_ticker columns possibly coming from multiple sources
    yf_cols = [c for c in ["yf_ticker", "yf_ticker_x", "yf_ticker_y"] if c in df.columns]
    if len(yf_cols) > 1:
        # Prefer non-null from the left-most, then coalesce
        base = df[yf_cols[0]]
        for c in yf_cols[1:]:
            base = base.fillna(df[c])
        df["yf_ticker"] = base
        # Drop extra suffixed cols
        for c in yf_cols:
            if c != "yf_ticker":
                df.drop(columns=[c], inplace=True)
    elif len(yf_cols) == 1 and yf_cols[0] != "yf_ticker":
        df.rename(columns={yf_cols[0]: "yf_ticker"}, inplace=True)

    # Fill yf_ticker from ticker when missing
    if "yf_ticker" in df.columns and "ticker" in df.columns:
        df["yf_ticker"] = df["yf_ticker"].fillna(df["ticker"])

    # Ensure English versions for title/content when language is English
    if "language" in df.columns:
        is_en = df["language"].astype(str).str.lower().isin(["en", "english", "eng"])
        # Title EN
        if "title_en" not in df.columns:
            df["title_en"] = None
        df.loc[is_en & (df["title_en"].isna() if "title_en" in df.columns else True), "title_en"] = df.get("title")
        # Content EN
        if "content_en" not in df.columns:
            df["content_en"] = None
        df.loc[is_en & (df["content_en"].isna() if "content_en" in df.columns else True), "content_en"] = df.get("content")

    # Compute actual_side from price_change_percentage
    if "price_change_percentage" in df.columns:
        def side_from_change(x):
            try:
                if pd.isna(x):
                    return None
                if x > 0:
                    return "up"
                if x < 0:
                    return "down"
                return "neutral"
            except Exception:
                return None
        df["actual_side"] = df["price_change_percentage"].apply(side_from_change)

    # Compute nextday_side from next-day price change percentage if available
    nextday_candidates = [
        "nextday_price_change_percentage",
        "next_day_price_change_percentage",
        "nextday_price_change_pct",
        "next_day_price_change_pct",
    ]
    nextday_col = first_existing_column(nextday_candidates)
    if nextday_col:
        def nextday_side_from_change(x):
            try:
                if pd.isna(x):
                    return None
                if x > 0:
                    return "up"
                if x < 0:
                    return "down"
                return "neutral"
            except Exception:
                return None
        df["nextday_side"] = df[nextday_col].apply(nextday_side_from_change)

    # Save enriched dataset
    enriched_output_path = "data/price_moves_norm_enriched.csv"
    df.to_csv(enriched_output_path, index=False)
    logger.info(f"Enriched data saved to '{enriched_output_path}'")

    # Drop rows with any missing values for a cleaned version
    cleaned_df = df.dropna()
    cleaned_output_path = "data/price_moves_norm_cleaned.csv"
    cleaned_df.to_csv(cleaned_output_path, index=False)
    logger.info(f"Cleaned data saved to '{cleaned_output_path}' (rows={len(cleaned_df)})")

    # Cleaning metrics
    cleaning_metrics_dir = os.path.join("data", "quality_metrics")
    os.makedirs(cleaning_metrics_dir, exist_ok=True)
    cleaning_metrics_path = os.path.join(cleaning_metrics_dir, "cleaning_metrics.csv")
    dropped_rows = len(df) - len(cleaned_df)
    metrics_df = pd.DataFrame([
        {
            "total_rows": len(df),
            "cleaned_rows": len(cleaned_df),
            "dropped_rows": dropped_rows,
            "dropped_pct": round((dropped_rows / len(df) * 100), 2) if len(df) else 0.0,
        }
    ])
    metrics_df.to_csv(cleaning_metrics_path, index=False)
    logger.info(f"Cleaning metrics saved to '{cleaning_metrics_path}'")
    
    # Log some statistics
    logger.info(f"Unique events: {df['event'].nunique()}")
    logger.info(f"Unique companies: {df['company'].nunique()}")
    logger.info(f"Date range: {df['published_date'].min()} to {df['published_date'].max()}")

    # Generate data quality reports
    quality_dir = os.path.join("data", "quality_metrics")
    os.makedirs(quality_dir, exist_ok=True)

    # Missing values per column
    missing_counts = df.isna().sum()
    missing_pct = (missing_counts / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        "column": missing_counts.index,
        "missing_count": missing_counts.values,
        "missing_pct": missing_pct.values,
    })
    missing_path = os.path.join(quality_dir, "missing_values.csv")
    missing_df.to_csv(missing_path, index=False)
    logger.info(f"Missing values report saved to '{missing_path}'")

    # Column uniqueness (distinct values per column)
    nunique = df.nunique(dropna=False)
    uniqueness_pct = (nunique / len(df) * 100).round(2)
    uniqueness_df = pd.DataFrame({
        "column": nunique.index,
        "unique_values": nunique.values,
        "total_rows": len(df),
        "uniqueness_pct": uniqueness_pct.values,
    })
    uniqueness_path = os.path.join(quality_dir, "column_uniqueness.csv")
    uniqueness_df.to_csv(uniqueness_path, index=False)
    logger.info(f"Column uniqueness report saved to '{uniqueness_path}'")

    # Full-row duplicates
    duplicated_mask = df.duplicated(keep=False)
    duplicate_rows = df[duplicated_mask]
    duplicate_rows_path = os.path.join(quality_dir, "duplicate_rows.csv")
    duplicate_rows.to_csv(duplicate_rows_path, index=False)
    logger.info(
        f"Duplicate rows report saved to '{duplicate_rows_path}' (count={len(duplicate_rows)})"
    )

    # Duplicates by news_id if available
    duplicates_by_news_id_path = None
    duplicate_news_id_groups = 0
    if "news_id" in df.columns:
        dup_news = df[df.duplicated(subset=["news_id"], keep=False)].sort_values(
            by=["news_id"]
        )
        duplicates_by_news_id_path = os.path.join(quality_dir, "duplicates_by_news_id.csv")
        dup_news.to_csv(duplicates_by_news_id_path, index=False)
        duplicate_news_id_groups = int((df["news_id"].value_counts() > 1).sum())
        logger.info(
            f"Duplicates by news_id saved to '{duplicates_by_news_id_path}' (groups={duplicate_news_id_groups})"
        )

    # Dataset-level summary JSON
    overall_missing = int(df.isna().sum().sum())
    overall_cells = int(len(df) * df.shape[1]) if len(df) > 0 else 0
    summary = {
        "num_rows": int(len(df)),
        "num_columns": int(df.shape[1]),
        "duplicate_row_count": int(duplicated_mask.sum()),
        "duplicate_row_pct": float(round(duplicated_mask.mean() * 100, 2)) if len(df) else 0.0,
        "overall_missing_count": overall_missing,
        "overall_missing_pct": float(round((overall_missing / overall_cells) * 100, 2)) if overall_cells else 0.0,
        "memory_usage_bytes": int(df.memory_usage(deep=True).sum()),
        "top_missing_columns": missing_df.sort_values("missing_pct", ascending=False)
        .head(10)
        .to_dict(orient="records"),
        "top_unique_columns": uniqueness_df.sort_values("uniqueness_pct", ascending=False)
        .head(10)
        .to_dict(orient="records"),
        "duplicates_by_news_id_file": duplicates_by_news_id_path,
        "duplicate_news_id_groups": duplicate_news_id_groups,
    }
    if "published_date" in df.columns:
        try:
            min_date = df["published_date"].min()
            max_date = df["published_date"].max()
            summary["published_date_range"] = {
                "min": None if pd.isna(min_date) else str(min_date),
                "max": None if pd.isna(max_date) else str(max_date),
            }
        except Exception:
            summary["published_date_range"] = None

    summary_path = os.path.join(quality_dir, "dataset_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"Dataset summary saved to '{summary_path}'")
    
    return df

if __name__ == "__main__":
    main() 