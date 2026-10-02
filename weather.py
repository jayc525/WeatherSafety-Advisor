"""
Open-Meteo API client for geocoding and weather data.
"""

import requests

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Fetch all useful fields so SOPs can check them without modifying this file.
CURRENT_FIELDS = [
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "precipitation",
    "precipitation_probability",
    "rain",
    "showers",
    "snowfall",
    "weather_code",
    "cloud_cover",
    "wind_speed_10m",
    "wind_gusts_10m",
    "wind_direction_10m",
    "uv_index",
    "is_day",
]

HOURLY_FIELDS = [
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "precipitation_probability",
    "precipitation",
    "rain",
    "showers",
    "snowfall",
    "weather_code",
    "cloud_cover",
    "wind_speed_10m",
    "wind_gusts_10m",
    "uv_index",
    "visibility",
]

# WMO Weather interpretation codes
# https://open-meteo.com/en/docs#weathervariables
WMO_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def geocode_city(city_name: str) -> dict | None:
    """
    Resolve a city name to coordinates using Open-Meteo geocoding API.
    Returns the top result or None if resolution fails.
    """
    try:
        resp = requests.get(
            GEOCODING_URL,
            params={"name": city_name, "count": 5, "language": "en"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results")
        if not results:
            return None

        top = results[0]
        return {
            "latitude": top["latitude"],
            "longitude": top["longitude"],
            "name": top.get("name", city_name),
            "country": top.get("country", ""),
            "admin1": top.get("admin1", ""),
        }
    except Exception:
        return None


def fetch_weather(latitude: float, longitude: float) -> dict | None:
    """
    Fetch current + hourly weather data from Open-Meteo.
    Returns the full API response or None on failure.
    """
    try:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": ",".join(CURRENT_FIELDS),
            "hourly": ",".join(HOURLY_FIELDS),
            "forecast_days": 1,
            "timezone": "auto",
        }
        resp = requests.get(FORECAST_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if "current" not in data:
            return None

        # Enrich with human-readable weather description
        code = data["current"].get("weather_code")
        data["current"]["weather_description"] = WMO_CODES.get(code, "Unknown")

        # Also enrich hourly weather codes
        if "hourly" in data and "weather_code" in data["hourly"]:
            data["hourly"]["weather_description"] = [
                WMO_CODES.get(c, "Unknown") for c in data["hourly"]["weather_code"]
            ]

        return data
    except Exception:
        return None


def format_weather_summary(weather_data: dict) -> str:
    """
    Format current weather into a clean summary string.
    Used when presenting weather data to the LLM or user.
    """
    current = weather_data.get("current", {})
    units = weather_data.get("current_units", {})

    lines = []
    field_labels = {
        "temperature_2m": "Temperature",
        "apparent_temperature": "Feels Like",
        "relative_humidity_2m": "Humidity",
        "precipitation": "Precipitation",
        "precipitation_probability": "Rain Probability",
        "wind_speed_10m": "Wind Speed",
        "wind_gusts_10m": "Wind Gusts",
        "uv_index": "UV Index",
        "cloud_cover": "Cloud Cover",
        "weather_description": "Conditions",
    }

    for field, label in field_labels.items():
        value = current.get(field)
        if value is not None:
            unit = units.get(field, "")
            if field == "weather_description":
                lines.append(f"  {label}: {value}")
            else:
                lines.append(f"  {label}: {value}{unit}")

    return "\n".join(lines)
