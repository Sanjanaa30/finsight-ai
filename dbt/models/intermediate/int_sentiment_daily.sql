-- Intermediate: daily news aggregates per category.
--
-- Phase 3 version is VOLUME-based: article_count per category per day. The
-- avg_sentiment column is a typed placeholder (null) that Phase 4 will fill once
-- FinBERT has scored each article. Building the structure now keeps the mart and
-- downstream models stable; only the values change later.

with news as (
    select * from {{ ref('stg_news') }}
),

daily as (
    select
        published_date as date,
        category,
        count(*) as article_count,
        cast(null as double) as avg_sentiment  -- filled in Phase 4 (FinBERT)
    from news
    group by published_date, category
)

select * from daily
