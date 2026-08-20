"""
Scan a slate for the validated ATS edge:
HOME UNDERDOG vs a ROAD FAVORITE of 7+ points.
(Held up out-of-sample: ~60-67% ATS, 2021-2025.)
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.ingest_odds import pull_game_odds

TARGET = sys.argv[1] if len(sys.argv) > 1 else "2026-09-13"


def main():
    odds = pull_game_odds(markets="spreads")
    if odds.empty:
        print("No spread odds available.")
        return

    ct = pd.to_datetime(odds["commence_time"], utc=True, errors="coerce")
    odds["et_date"] = (ct - pd.Timedelta(hours=4)).dt.date.astype(str)
    slate = odds[odds["et_date"] == TARGET]
    spreads = slate[slate["market"] == "spreads"]

    print("=" * 74)
    print(f"HOME-DOG EDGE SCAN — {TARGET}")
    print("Target: road team favored by 7+  ->  bet the HOME underdog")
    print("=" * 74)

    qualifying = []
    seen = set()
    for gid, grp in spreads.groupby("game_id"):
        home = grp["home_team"].iloc[0]
        away = grp["away_team"].iloc[0]
        # home team's spread point
        home_row = grp[grp["outcome_name"] == home]
        if home_row.empty:
            continue
        home_pt = home_row["outcome_point"].iloc[0]
        # home_pt >= 7 means home getting 7+ points = road team favored by 7+
        matchup = f"{away} @ {home}"
        tag = ""
        if home_pt >= 7:
            tag = "  <-- QUALIFIES (home dog +%.1f)" % home_pt
            qualifying.append((matchup, home, home_pt))
        print(f"  {matchup:<44} home {home_pt:+.1f}{tag}")

    print("\n" + "=" * 74)
    if qualifying:
        print(f"QUALIFYING HOME-DOG PLAYS: {len(qualifying)}")
        for matchup, home, pt in qualifying:
            print(f"  Bet {home} +{pt:.1f}  ({matchup})")
    else:
        print("No games this slate have a road favorite of 7+. No home-dog play.")


if __name__ == "__main__":
    main()
