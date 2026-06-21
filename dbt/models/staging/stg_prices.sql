-- Staging: cleaned daily OHLCV. One row per ticker per trading day.
-- Light cleaning only: cast the timestamp to a date, drop rows with no close,
-- and de-duplicate on (ticker, date) keeping the most recently ingested row.

with source as (
    select * from {{ source('raw', 'prices') }}
),

cleaned as (
    select
        cast(date as date) as date,
        ticker,
        open,
        high,
        low,
        close,
        volume,
        ingested_at
    from source
    where close is not null
)

select *
from cleaned
qualify row_number() over (partition by ticker, date order by ingested_at desc) = 1
