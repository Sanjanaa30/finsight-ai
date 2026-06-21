-- Staging: cleaned news articles, still tagged by category. Adds a published_date
-- (for daily aggregation downstream), drops rows with no title, and removes
-- duplicate/syndicated repeats within a category (same title + publish time).

with source as (
    select * from {{ source('raw', 'news') }}
),

cleaned as (
    select
        category,
        source as source_name,
        author,
        title,
        description,
        url,
        published_at,
        cast(published_at as date) as published_date,
        content,
        ingested_at
    from source
    where title is not null
)

select *
from cleaned
qualify row_number() over (
    partition by category, title, published_at
    order by ingested_at desc
) = 1
