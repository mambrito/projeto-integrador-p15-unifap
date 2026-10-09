CREATE TABLE IF NOT EXISTS bike_daily (
 dteday DATE PRIMARY KEY,
 holiday INTEGER NOT NULL CHECK (holiday IN (0,1)),
 workingday INTEGER NOT NULL CHECK (workingday IN (0,1)),
 cnt INTEGER NOT NULL CHECK (cnt >= 0)
);
CREATE TABLE IF NOT EXISTS weather_daily (
 dteday DATE PRIMARY KEY,
 temperature_2m_mean DOUBLE PRECISION NOT NULL,
 precipitation_sum DOUBLE PRECISION NOT NULL CHECK (precipitation_sum >= 0),
 wind_speed_10m_max DOUBLE PRECISION NOT NULL CHECK (wind_speed_10m_max >= 0)
);
CREATE OR REPLACE VIEW integrated_daily AS
 SELECT b.*, w.temperature_2m_mean, w.precipitation_sum, w.wind_speed_10m_max
 FROM bike_daily b JOIN weather_daily w USING (dteday);
