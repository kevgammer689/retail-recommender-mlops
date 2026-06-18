CREATE TABLE IF NOT EXISTS recommendation_logs (
    id BIGSERIAL PRIMARY KEY,
    request_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    cart_product_ids JSONB NOT NULL,
    top_k INTEGER NOT NULL,
    recommendation_count INTEGER NOT NULL,
    recommended_product_ids JSONB NOT NULL,
    model_type TEXT NOT NULL,
    serving_mode TEXT NOT NULL,
    latency_ms NUMERIC NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_recommendation_logs_created_at
    ON recommendation_logs (created_at);

CREATE INDEX IF NOT EXISTS idx_recommendation_logs_request_id
    ON recommendation_logs (request_id);
