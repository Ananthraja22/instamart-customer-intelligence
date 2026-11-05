-- Instamart analytical queries (SQLite 3.25+). Only completed orders count as revenue or spend.
-- Each query starts with a "-- Qn:" header; scripts/a4_sql.py splits this file on those headers.

-- Q1: WHERE filters rows before grouping; HAVING filters the groups afterwards.
SELECT customer_id,
       COUNT(*)                    AS orders_over_500,
       ROUND(SUM(order_amount), 2) AS total_inr
FROM orders
WHERE status = 'completed'
  AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total_inr DESC;

-- Q2: Revenue by city and product category (three-table join).
SELECT c.city,
       p.category,
       ROUND(SUM(o.order_amount), 2) AS revenue_inr
FROM orders o
JOIN customers c ON c.customer_id = o.customer_id
JOIN products  p ON p.product_id  = o.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY revenue_inr DESC;

-- Q3: Products priced above the average catalogue price (scalar subquery).
SELECT product_id, product_name, category, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC;

-- Q4: Customers with no completed order. The status test belongs in ON: in WHERE it would
--     discard the unmatched rows and turn the LEFT JOIN into an inner join.
SELECT c.customer_id, c.city
FROM customers c
LEFT JOIN orders o
       ON o.customer_id = c.customer_id
      AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;

-- Q5: Customers whose total completed spend is above the average of those totals (two CTEs).
WITH customer_totals AS (
    SELECT customer_id, SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
average_total AS (
    SELECT AVG(total_spend) AS avg_spend
    FROM customer_totals
)
SELECT t.customer_id,
       ROUND(t.total_spend, 2) AS total_spend,
       ROUND(a.avg_spend, 2)   AS avg_spend
FROM customer_totals t
CROSS JOIN average_total a
WHERE t.total_spend > a.avg_spend
ORDER BY t.total_spend DESC;

-- Q6: Rank customers by spend within their city. Totals are rounded to paise first, so two equal
--     totals tie instead of differing by floating-point noise.
WITH customer_totals AS (
    SELECT c.city,
           o.customer_id,
           ROUND(SUM(o.order_amount), 2) AS total_spend
    FROM orders o
    JOIN customers c ON c.customer_id = o.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.city, o.customer_id
)
SELECT city,
       customer_id,
       total_spend,
       RANK()       OVER (PARTITION BY city ORDER BY total_spend DESC) AS spend_rank,
       DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS spend_dense_rank,
       ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS spend_row_number
FROM customer_totals
ORDER BY city, spend_row_number;

-- Q7: Daily completed revenue with a running total.
WITH daily AS (
    SELECT SUBSTR(order_ts, 1, 10) AS day,
           SUM(order_amount)       AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY day
)
SELECT day,
       ROUND(revenue, 2)                             AS revenue_inr,
       ROUND(SUM(revenue) OVER (ORDER BY day), 2)    AS cumulative_revenue_inr
FROM daily
ORDER BY day;
