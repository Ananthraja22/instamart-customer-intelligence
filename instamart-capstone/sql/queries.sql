-- Run these queries against state/instamart.db using SQLite.

-- 1. Order totals by status
SELECT
    status,
    COUNT(*) AS order_count,
    ROUND(SUM(order_amount), 2) AS total_order_amount,
    ROUND(AVG(order_amount), 2) AS average_order_amount
FROM orders
GROUP BY status
ORDER BY order_count DESC;

-- 2. Monthly order volume and amounts across all statuses
SELECT
    substr(order_ts, 1, 7) AS order_month,
    COUNT(*) AS order_count,
    ROUND(SUM(order_amount), 2) AS total_order_amount_inr,
    ROUND(AVG(order_amount), 2) AS average_order_amount
FROM orders
GROUP BY order_month
ORDER BY order_month;

-- 3. Completed sales and units by product category
SELECT
    p.category,
    COUNT(*) AS completed_order_count,
    SUM(o.quantity) AS units_sold,
    ROUND(SUM(o.order_amount), 2) AS sales_inr
FROM orders AS o
JOIN products AS p ON p.product_id = o.product_id
WHERE lower(o.status) = 'completed'
GROUP BY p.category
ORDER BY sales_inr DESC;

-- 4. Top 20 customers by total order amount
SELECT
    c.customer_id,
    c.city,
    COUNT(o.order_id) AS order_count,
    ROUND(SUM(o.order_amount), 2) AS spend_inr
FROM customers AS c
JOIN orders AS o ON o.customer_id = c.customer_id
WHERE lower(o.status) = 'completed'
GROUP BY c.customer_id, c.city
ORDER BY spend_inr DESC
LIMIT 20;

-- 5. Completed orders delivered after 45 minutes
-- This is a potential refund pool, before checking request date and approvals.
SELECT
    COUNT(*) AS eligible_late_orders,
    ROUND(SUM(order_amount), 2) AS potential_refund_inr,
    ROUND(AVG(delivery_minutes), 1) AS average_delivery_minutes
FROM orders
WHERE lower(status) = 'completed'
  AND delivery_minutes > 45;

-- 6. Ticket counts by category
SELECT
    lower(true_category) AS category,
    COUNT(*) AS ticket_count
FROM tickets
GROUP BY lower(true_category)
ORDER BY ticket_count DESC;
