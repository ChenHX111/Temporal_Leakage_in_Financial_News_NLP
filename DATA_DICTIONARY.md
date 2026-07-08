# Data dictionary — classifier_training_v2.parquet

56409 rows x 51 columns.

| column | dtype | description |
|---|---|---|
| `news_id` | int64 | unique article id |
| `yf_ticker` | str | instrument ticker (yfinance) |
| `exchange` | str | listing exchange |
| `etf_ticker` | str | engineered feature (keyword/timing/sentiment/length) |
| `market_status` | str | pre/regular/after-hours at publication |
| `title_en` | str | article title (English) |
| `content_en` | str | article body (English) |
| `event` | str | event-type tag (203 types) |
| `publisher` | str | news publisher |
| `published_date` | str | publication timestamp |
| `industry` | str | issuer industry |
| `publisher_topic` | str | engineered feature (keyword/timing/sentiment/length) |
| `price_change_percentage` | float64 | same/next-day return % of the instrument |
| `price_change` | float64 | engineered feature (keyword/timing/sentiment/length) |
| `index_price_change` | float64 | engineered feature (keyword/timing/sentiment/length) |
| `index_price_change_percentage` | float64 | benchmark index return % |
| `actual_side` | str | label: up/down/neutral (return direction) |
| `nextday_side` | str | next-day direction label |
| `nextday_price_change_percentage` | float64 | next-day return % |
| `is_holiday_period` | int64 | engineered feature (keyword/timing/sentiment/length) |
| `day_of_week` | int64 | engineered feature (keyword/timing/sentiment/length) |
| `day_of_month` | int64 | engineered feature (keyword/timing/sentiment/length) |
| `day_of_year` | int64 | engineered feature (keyword/timing/sentiment/length) |
| `financial_keyword_count` | int64 | engineered feature (keyword/timing/sentiment/length) |
| `content_length` | int64 | engineered feature (keyword/timing/sentiment/length) |
| `has_percentages` | int64 | engineered feature (keyword/timing/sentiment/length) |
| `year` | int64 | engineered feature (keyword/timing/sentiment/length) |
| `regulatory_keyword_count` | int64 | engineered feature (keyword/timing/sentiment/length) |
| `size_keyword_count` | int64 | engineered feature (keyword/timing/sentiment/length) |
| `market_keyword_count` | int64 | engineered feature (keyword/timing/sentiment/length) |
| `avg_sentence_length` | float64 | engineered feature (keyword/timing/sentiment/length) |
| `is_earnings_season` | int64 | engineered feature (keyword/timing/sentiment/length) |
| `urgency_keyword_count` | int64 | engineered feature (keyword/timing/sentiment/length) |
| `is_high_volatility_period` | int64 | engineered feature (keyword/timing/sentiment/length) |
| `period_count` | int64 | engineered feature (keyword/timing/sentiment/length) |
| `week_of_year` | int64 | engineered feature (keyword/timing/sentiment/length) |
| `month` | int64 | engineered feature (keyword/timing/sentiment/length) |
| `title_length` | int64 | engineered feature (keyword/timing/sentiment/length) |
| `has_quotes` | int64 | engineered feature (keyword/timing/sentiment/length) |
| `is_month_end` | int64 | engineered feature (keyword/timing/sentiment/length) |
| `is_market_hours` | int64 | engineered feature (keyword/timing/sentiment/length) |
| `sentence_count` | int64 | engineered feature (keyword/timing/sentiment/length) |
| `hour` | int64 | engineered feature (keyword/timing/sentiment/length) |
| `sentiment_subjectivity` | float64 | engineered feature (keyword/timing/sentiment/length) |
| `comma_count` | int64 | engineered feature (keyword/timing/sentiment/length) |
| `content_word_count` | int64 | engineered feature (keyword/timing/sentiment/length) |
| `avg_word_length` | float64 | engineered feature (keyword/timing/sentiment/length) |
| `sentiment_polarity` | float64 | engineered feature (keyword/timing/sentiment/length) |
| `is_pre_market` | int64 | engineered feature (keyword/timing/sentiment/length) |
| `title_word_count` | int64 | engineered feature (keyword/timing/sentiment/length) |
| `created_at` | datetime64[us] | engineered feature (keyword/timing/sentiment/length) |