CREATE TABLE IF NOT EXISTS departments (
    department_id INTEGER PRIMARY KEY,
    department VARCHAR(255) NOT NULL
);

CREATE TABLE IF NOT EXISTS aisles (
    aisle_id INTEGER PRIMARY KEY,
    aisle VARCHAR(255) NOT NULL
);

CREATE TABLE IF NOT EXISTS products (
    product_id INTEGER PRIMARY KEY,
    product_name VARCHAR(255) NOT NULL,
    aisle_id INTEGER NOT NULL,
    department_id INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    order_id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    eval_set VARCHAR(20) NOT NULL,
    order_number INTEGER NOT NULL,
    order_dow INTEGER NOT NULL,
    order_hour_of_day INTEGER NOT NULL,
    days_since_prior_order NUMERIC
);

CREATE TABLE IF NOT EXISTS order_products_prior (
    order_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    add_to_cart_order INTEGER NOT NULL,
    reordered INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS order_products_train (
    order_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    add_to_cart_order INTEGER NOT NULL,
    reordered INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_products_aisle_id
    ON products (aisle_id);

CREATE INDEX IF NOT EXISTS idx_products_department_id
    ON products (department_id);

CREATE INDEX IF NOT EXISTS idx_orders_user_id
    ON orders (user_id);

CREATE INDEX IF NOT EXISTS idx_orders_eval_set
    ON orders (eval_set);

CREATE INDEX IF NOT EXISTS idx_order_products_prior_order_id
    ON order_products_prior (order_id);

CREATE INDEX IF NOT EXISTS idx_order_products_prior_product_id
    ON order_products_prior (product_id);

CREATE INDEX IF NOT EXISTS idx_order_products_train_order_id
    ON order_products_train (order_id);

CREATE INDEX IF NOT EXISTS idx_order_products_train_product_id
    ON order_products_train (product_id);
