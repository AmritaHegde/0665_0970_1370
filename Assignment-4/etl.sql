CREATE DATABASE ETL;
CREATE TABLE user_device
(
use_id INT PRIMARY KEY,
user_id INT,
platform VARCHAR(50),
platform_version DECIMAL(10,3),
device VARCHAR(50),
use_type_id INT
);

CREATE TABLE android_devices(
Retail_branding VARCHAR(100) NULL,
Marketing_Name VARCHAR(100) NULL,
Device VARCHAR(40),
Model VARCHAR(100)
);