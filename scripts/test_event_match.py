import requests

API_KEY = "1c2e7d0377ac3dd72171dc52a8382260"
url = "https://api.the-odds-api.com/v4/historical/sports/americanfootball_nfl/events"
params = {"apiKey": API_KEY, "date": "2025-09-07T18:00:00Z"}
r = requests.get(url, params=params, timeout=30)
data = r.json()
events = data.get("data", [])
print(f"Found {len(events)} events")
for e in events[:5]:
    print(f"  ID: {e['id']} | {e['away_team']} @ {e['home_team']} | {e['commence_time']}")
