-- Staging: cleaned macro indicators in long form. Casts the date, drops
-- unreleased (null) observations, and de-duplicates on (series_id, date).

with source as (
    select * from {{ source('raw', 'macro') }}
),

cleaned as (
    select
        series_id,
        series_name,
        cast(date as date) as date,
        value,
        ingested_at
    from source
    where value is not null
)

select *
from cleaned
qualify row_number() over (partition by series_id, date order by ingested_at desc) = 1
