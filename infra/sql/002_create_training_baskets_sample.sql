-- Crea una tabla derivada para entrenamiento inicial del recomendador.
-- Esta tabla reduce el tamaño del dataset para poder iterar rápido en local.
-- Usamos solo órdenes del eval_set = 'prior', porque representan historial de compras.

DROP TABLE IF EXISTS training_baskets_sample;

CREATE TABLE training_baskets_sample AS
WITH sampled_orders AS (
    SELECT
        order_id,
        user_id,
        order_number,
        order_dow,
        order_hour_of_day,
        days_since_prior_order
    FROM orders
    WHERE eval_set = 'prior'
    ORDER BY order_id
    LIMIT 100000
)
SELECT
    so.order_id,
    so.user_id,
    so.order_number,
    so.order_dow,
    so.order_hour_of_day,
    so.days_since_prior_order,
    op.product_id,
    op.add_to_cart_order,
    op.reordered,
    p.product_name,
    p.aisle_id,
    a.aisle,
    p.department_id,
    d.department
FROM sampled_orders so
INNER JOIN order_products_prior op
    ON so.order_id = op.order_id
INNER JOIN products p
    ON op.product_id = p.product_id
INNER JOIN aisles a
    ON p.aisle_id = a.aisle_id
INNER JOIN departments d
    ON p.department_id = d.department_id;

-- Índices para acelerar consultas de entrenamiento.
CREATE INDEX idx_training_baskets_sample_order_id
    ON training_baskets_sample (order_id);

CREATE INDEX idx_training_baskets_sample_product_id
    ON training_baskets_sample (product_id);

CREATE INDEX idx_training_baskets_sample_user_id
    ON training_baskets_sample (user_id);