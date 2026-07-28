import requests

url = "https://api.the-odds-api.com/v4/historical/sports/americanfootball_nfl/odds"
params = {
    "apiKey": "1c2e7d0377ac3dd72171dc52a8382260",
    "regions": "us",
    "markets": "h2h",
    "date": "2025-09-05T12:00:00Z",
}

r = requests.get(url, params=params, timeout=30)
print(f"Status: {r.status_code}")
print(f"Remaining: {r.headers.get('x-requests-remaining', '?')}")
print(r.text[:300])
