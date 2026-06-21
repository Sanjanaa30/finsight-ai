-- Intermediate: per-ticker price features for ML.
-- Uses SQL window functions over stg_prices:
--   daily_return   : 1-day percentage change in close
--   ma20 / ma50    : 20- and 50-day moving averages of close
--   volatility_20d : rolling 20-day standard deviation of daily return
-- Early rows have partial windows (fewer than N prior days) by design.

with prices as (
    select * from {{ ref('stg_prices') }}
),

returns as (
    select
        date,
        ticker,
        close,
        volume,
        close / lag(close) over (partition by ticker order by date) - 1 as daily_return
    from prices
),

features as (
    select
        date,
        ticker,
        close,
        volume,
        daily_return,
        avg(close) over (
            partition by ticker order by date
            rows between 19 preceding and current row
        ) as ma20,
        avg(close) over (
            partition by ticker order by date
            rows between 49 preceding and current row
        ) as ma50,
        stddev_samp(daily_return) over (
            partition by ticker order by date
            rows between 19 preceding and current row
        ) as volatility_20d
    from returns
)

select * from features
