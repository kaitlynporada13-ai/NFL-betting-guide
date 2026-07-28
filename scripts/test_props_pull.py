import requests

API_KEY = "1c2e7d0377ac3dd72171dc52a8382260"

# Test pulling props for a specific event
event_id = "e2d3b0596efcebc1c19dab92009f5e26"  # Bengals @ Browns 2025-09-07
url = f"https://api.the-odds-api.com/v4/historical/sports/americanfootball_nfl/events/{event_id}/odds"
params = {
    "apiKey": API_KEY,
    "regions": "us",
    "markets": "player_pass_yds,player_rush_yds,player_reception_yds",
    "bookmakers": "fanduel",
    "date": "2025-09-07T15:00:00Z",
    "oddsFormat": "american",
}

r = requests.get(url, params=params, timeout=30)
print(f"Status: {r.status_code}")
print(f"Credits remaining: {r.headers.get('x-requests-remaining', '?')}")

if r.status_code == 200:
    data = r.json()
    event = data.get("data", {})
    bookmakers = event.get("bookmakers", [])
    print(f"Bookmakers returned: {len(bookmakers)}")
    for bk in bookmakers:
        print(f"  {bk['key']}: {len(bk.get('markets',[]))} markets")
        for mkt in bk.get("markets", []):
            outcomes = mkt.get("outcomes", [])
            print(f"    {mkt['key']}: {len(outcomes)} outcomes")
            for o in outcomes[:4]:
                print(f"      {o.get('description','?')} {o.get('name')} {o.get('point')} @ {o.get('price')}")
else:
    print(r.text[:300])
