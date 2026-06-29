CREATE TABLE IF NOT EXISTS experiments (
    id              TEXT PRIMARY KEY,  
    timestamp       TEXT NOT NULL,  
    benchmark_name  TEXT NOT NULL,  
    estimator_name  TEXT NOT NULL,  
    random_seed     INTEGER NOT NULL,  
    status          TEXT NOT NULL  
);  
  
CREATE TABLE IF NOT EXISTS metrics (
    experiment_id       TEXT NOT NULL,
    rmse_per_dim        TEXT,   -- JSON object {state_var: rmse}; no pooled scalar
    runtime_seconds     REAL,
    runtime_per_step_ms REAL,
    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
);
  
CREATE TABLE IF NOT EXISTS artifacts (  
    experiment_id   TEXT NOT NULL,  
    model_path      TEXT,  
    figure_path     TEXT,  
    FOREIGN KEY (experiment_id) REFERENCES experiments (id)  
);
