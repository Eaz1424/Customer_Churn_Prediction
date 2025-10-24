-- Customer Churn Prediction Model: SQL import and analytics helpers
-- Adjust schema/datatype details to match your production warehouse (e.g., Snowflake, BigQuery, Redshift).

CREATE TABLE IF NOT EXISTS churn_customers (
    customer_id                  VARCHAR(16) PRIMARY KEY,
    orders_last_3_months         SMALLINT NOT NULL,
    avg_order_value              NUMERIC(10, 2) NOT NULL,
    discount_usage               SMALLINT,
    days_since_last_order        SMALLINT,
    customer_tenure              SMALLINT,
    membership_tier              VARCHAR(20),
    complaints_filed             SMALLINT,
    email_open_rate              NUMERIC(5, 2),
    churn                        SMALLINT NOT NULL,
    predicted_churn_probability  NUMERIC(6, 4),
    predicted_label              SMALLINT,
    imported_at                  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Bulk import the dashboard-ready dataset created by the notebook (adjust path for your environment).
-- COPY commands vary across warehouses; examples for PostgreSQL and Snowflake are provided.

-- PostgreSQL example:
-- COPY churn_customers (
--     customer_id,
--     orders_last_3_months,
--     avg_order_value,
--     discount_usage,
--     days_since_last_order,
--     customer_tenure,
--     membership_tier,
--     complaints_filed,
--     email_open_rate,
--     churn,
--     predicted_churn_probability
-- )
-- FROM '/absolute/path/to/data/dashboard_customer_churn_view.csv'
-- WITH (FORMAT csv, HEADER true);

-- Snowflake example:
-- COPY INTO churn_customers (
--     customer_id,
--     orders_last_3_months,
--     avg_order_value,
--     discount_usage,
--     days_since_last_order,
--     customer_tenure,
--     membership_tier,
--     complaints_filed,
--     email_open_rate,
--     churn,
--     predicted_churn_probability
-- )
-- FROM @~/staging/dashboard_customer_churn_view.csv
-- FILE_FORMAT = (TYPE = CSV FIELD_DELIMITER = ',' SKIP_HEADER = 1);

-- Example analytic query: prioritize segments for reactivation campaigns.
SELECT
    membership_tier,
    COUNT(*) AS customers,
    AVG(predicted_churn_probability) AS avg_churn_risk,
    AVG(CASE WHEN churn = 1 THEN 1 ELSE 0 END) AS historical_churn_rate,
    AVG(orders_last_3_months) AS avg_recent_orders,
    AVG(discount_usage) AS avg_discount_usage
FROM churn_customers
GROUP BY membership_tier
ORDER BY avg_churn_risk DESC;

-- Example retention list pull: high-risk customers with marketing opt-in inferred from email engagement.
SELECT
    customer_id,
    predicted_churn_probability,
    orders_last_3_months,
    discount_usage,
    days_since_last_order,
    membership_tier
FROM churn_customers
WHERE predicted_churn_probability >= 0.65
  AND COALESCE(email_open_rate, 0) >= 0.15
ORDER BY predicted_churn_probability DESC;
