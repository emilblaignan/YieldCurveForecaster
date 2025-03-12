# Configuration values for the yield curve forecasting project

# Database configuration
DB_FILE = "yields.db"

# Date ranges
DEFAULT_START_DATE = "2005-01-01"
DEFAULT_END_DATE = "2025-01-01"

# Available maturities (in years)
MATURITIES_LIST = [30, 10, 5, 2, 1, 0.5, 0.25, 0.0833]

# Forecast parameters
DEFAULT_FORECAST_STEPS = 24
FORECAST_HORIZONS = [1, 3, 6, 12]  # months

# Nelson-Siegel parameters
DEFAULT_LAMBDA = 0.0609