-- Mart: one wide, ML-ready row per ticker per trading day.
--
-- Combines three feature groups:
--   1. Price features (per ticker)      <- int_price_features
--   2. Macro context (market-wide)      <- stg_macro, attached via ASOF join so
--      each daily price row carries the latest macro value released on or before
--      that date (macro is monthly/quarterly; prices are daily).
--   3. News context (market-wide)       <- int_sentiment_daily, aggregated to a
--      daily total. news_avg_sentiment stays null until Phase 4 (FinBERT).
--
-- Macro/news columns are null for dates outside their coverage (e.g. news only
-- spans the last ~30 days) -- expected, handled downstream.

with px as (
    select * from {{ ref('int_price_features') }}
),

fed as (select date, value from {{ ref('stg_macro') }} where series_id = 'FEDFUNDS'),
cpi as (select date, value from {{ ref('stg_macro') }} where series_id = 'CPIAUCSL'),
unemp as (select date, value from {{ ref('stg_macro') }} where series_id = 'UNRATE'),
gdp as (select date, value from {{ ref('stg_macro') }} where series_id = 'GDP'),

news as (
    select
        date,
        sum(article_count) as news_article_count,
        avg(avg_sentiment) as news_avg_sentiment  -- null until Phase 4
    from {{ ref('int_sentiment_daily') }}
    group by date
)

select
    px.date,
    px.ticker,
    px.close,
    px.volume,
    px.daily_return,
    px.ma20,
    px.ma50,
    px.volatility_20d,
    fed.value as fed_funds_rate,
    cpi.value as cpi,
    unemp.value as unemployment_rate,
    gdp.value as gdp,
    news.news_article_count,
    news.news_avg_sentiment
from px
asof left join fed on px.date >= fed.date
asof left join cpi on px.date >= cpi.date
asof left join unemp on px.date >= unemp.date
asof left join gdp on px.date >= gdp.date
left join news on news.date = px.date
