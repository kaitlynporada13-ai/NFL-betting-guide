import requests, json
from pathlib import Path

API_KEY = "pmx_c5f09ac5dafd61ed894fc1adcfa26bd0"
BASE = "https://api.parse.bot/scraper/8cbb56cd-270c-41c6-ab5b-ff713cf1ef13"
OUT = Path("data/raw")

# Pull Rankings for all 4 positions (40 credits)
for pos in ["QB", "RB", "WR", "TE"]:
    r = requests.get(f"{BASE}/get_rankings", headers={"X-API-Key": API_KEY}, params={
        "depth": "rankings",
        "scoring": "half-ppr",
        "position": pos,
        "is_dynasty": "false",
        "league_type": "standard",
    }, timeout=30)
    print(f"Rankings {pos}: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        with open(OUT / f"draftsharks_rankings_{pos.lower()}.json", "w") as f:
            json.dump(data, f)
        players = data.get("data", {}).get("players", [])
        print(f"  Players: {len(players)}")
        if players:
            p = players[0]
            print(f"  Sample: {p.get('name')} | risk: {p.get('injury_risk')} | floor: {p.get('floor')} | ceil: {p.get('ceiling')}")
    else:
        print(f"  Error: {r.text[:200]}")
