"""
GAME SPECIALS engine.
FanDuel's bespoke specials/boosts are NOT in the odds API, and we have no VALIDATED
betting edge on them. What we CAN do honestly: compute historical BASE-RATE probabilities
for common special structures, and combine them across the games in a time window
(e.g., "every team in the 1pm window kicks a FG"). Compare our probability to FanDuel's
offered odds to judge value yourself.

Confidence here = how likely the event is (base rate), NOT a validated edge.
"""
import pandas as pd
from pathlib import Path
from datetime import date

from pipeline.config_loader import get_data_dir
from pipeline.ingest_odds import pull_game_odds

RAW = get_data_dir("raw")
PROC = get_data_dir("processed")


def base_rates():
    """Historical game-level base rates (2023-2025)."""
    g = pd.read_parquet(RAW / "games_historical.parquet")
    g = g[(g["season"] >= 2023) & g["home_score"].notna()].copy()

    # Both teams score (points > 0)
    both_score = ((g["home_score"] > 0) & (g["away_score"] > 0)).mean()

    # FG rate per team-game from player stats (fg_made column, new nflverse format)
    ps = pd.read_parquet(RAW / "player_stats_historical.parquet")
    rates = {"both_teams_score": both_score}
    if "fg_made" in ps.columns:
        ps2 = ps[ps["season"] >= 2023]
        # team-game fg made
        tg = ps2.groupby(["recent_team", "season", "week"])["fg_made"].sum().reset_index()
        rates["team_kicks_fg"] = (tg["fg_made"] >= 1).mean()
    else:
        rates["team_kicks_fg"] = 0.85  # league-typical fallback
    return rates


def current_week():
    ss = date(2026, 9, 10)
    t = date.today()
    if t < ss:
        return 1 if (ss - t).days <= 28 else 0
    return min(max(1, (t - ss).days // 7 + 1), 22)


def window_label(hour_et):
    if hour_et <= 13:
        return "1pm window"
    if hour_et <= 17:
        return "4pm window"
    return "primetime"


def build_specials():
    rates = base_rates()
    fg = rates["team_kicks_fg"]
    both = rates["both_teams_score"]

    odds = pull_game_odds(markets="totals")
    if odds.empty:
        return pd.DataFrame()
    games = odds[["game_id", "home_team", "away_team", "commence_time"]].drop_duplicates()
    ct = pd.to_datetime(games["commence_time"], utc=True, errors="coerce")
    games["hour_et"] = (ct - pd.Timedelta(hours=4)).dt.hour
    games["window"] = games["hour_et"].apply(window_label)

    rows = []

    # Per-window specials
    for window, grp in games.groupby("window"):
        n_teams = len(grp) * 2
        n_games = len(grp)

        # "Every team in the window kicks a FG" = fg ^ n_teams
        p_all_fg = fg ** n_teams
        rows.append({
            "special": f"Every team kicks a FG — {window}",
            "scope": f"{n_games} games / {n_teams} teams",
            "probability": round(p_all_fg * 100, 1),
            "confidence": conf_from_prob(p_all_fg),
            "why": (f"Each team kicks a FG ~{fg:.0%} of games historically. Across {n_teams} teams "
                    f"that compounds to ~{p_all_fg:.0%}. Long parlay — only bet if FanDuel's odds "
                    f"pay more than ~{int(1/p_all_fg)}-to-1."),
        })

        # "Both teams score in every game in window" = both ^ n_games
        p_all_both = both ** n_games
        rows.append({
            "special": f"Both teams score in every game — {window}",
            "scope": f"{n_games} games",
            "probability": round(p_all_both * 100, 1),
            "confidence": conf_from_prob(p_all_both),
            "why": (f"Both teams score in ~{both:.0%} of games. Across {n_games} games "
                    f"that's ~{p_all_both:.0%}. High-probability but low-payout parlay."),
        })

    # Single-game reference specials
    rows.append({
        "special": "Both teams score a FG (single game)", "scope": "per game",
        "probability": round((fg ** 2) * 100, 1), "confidence": conf_from_prob(fg ** 2),
        "why": f"Each team kicks a FG ~{fg:.0%}; both in the same game ~{fg**2:.0%} (assumes independence).",
    })

    df = pd.DataFrame(rows).sort_values("probability", ascending=False)
    df.to_parquet(PROC / "specials_picks_latest.parquet", index=False)
    return df


def conf_from_prob(p):
    if p >= 0.75:
        return "HIGH prob"
    if p >= 0.50:
        return "MEDIUM prob"
    if p >= 0.25:
        return "LOW prob"
    return "LONGSHOT"


def main():
    df = build_specials()
    if df.empty:
        print("No games available to build specials.")
        return
    print(f"Built {len(df)} slate specials (base-rate probabilities).")
    for _, r in df.iterrows():
        print(f"  [{r['confidence']:<12}] {r['probability']:>5.1f}%  {r['special']}")


if __name__ == "__main__":
    main()
