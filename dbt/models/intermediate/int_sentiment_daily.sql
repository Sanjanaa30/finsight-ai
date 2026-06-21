-- Intermediate: daily news aggregates per category, now with real FinBERT
-- sentiment (Phase 4). avg_sentiment is the mean P(positive)-P(negative) across
-- a category's articles that day, in [-1, 1]. article_count is the daily volume.

with scored as (
    select * from {{ ref('stg_news_scored') }}
),

daily as (
    select
        published_date as date,
        category,
        count(*) as article_count,
        avg(sentiment_score) as avg_sentiment
    from scored
    group by published_date, category
)

select * from daily
