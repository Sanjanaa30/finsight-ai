-- Staging: news articles with FinBERT sentiment scores (Phase 4 output).
-- Cleans the scored Parquet the same way stg_news cleans the raw feed:
-- drop rows with no title and de-duplicate syndicated repeats within a category.

with source as (
    select * from {{ source('processed', 'news_scored') }}
),

cleaned as (
    select
        category,
        source as source_name,
        title,
        url,
        cast(published_date as date) as published_date,
        published_at,
        sentiment_score,
        sentiment_label
    from source
    where title is not null
)

select
    category,
    source_name,
    title,
    url,
    published_date,
    sentiment_score,
    sentiment_label
from cleaned
qualify row_number() over (
    partition by category, title, published_at
    order by published_at desc
) = 1
