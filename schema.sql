CREATE DATABASE IF NOT EXISTS bizforecast;
USE bizforecast;

CREATE TABLE IF NOT EXISTS products (
  product_id VARCHAR(16) PRIMARY KEY,
  product_name VARCHAR(100) NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sales (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  product_id VARCHAR(16) NOT NULL,
  sale_date DATE NOT NULL,
  sales INT NOT NULL,
  FOREIGN KEY (product_id) REFERENCES products(product_id),
  UNIQUE KEY uniq_product_date (product_id, sale_date)
);

CREATE TABLE IF NOT EXISTS inventory (
  product_id VARCHAR(16) PRIMARY KEY,
  current_inventory INT NOT NULL,
  lead_time_days INT NOT NULL DEFAULT 7,
  service_level DOUBLE NOT NULL DEFAULT 0.95,
  target_days INT NOT NULL DEFAULT 14,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  FOREIGN KEY (product_id) REFERENCES products(product_id)
);