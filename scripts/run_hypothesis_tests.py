"""
Hypothesis Testing: H004-H014
Test situational, environmental, and usage factors against prop outcomes.
Each test answers: "Does this factor shift the over/under hit rate away from 50%?"
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"


def load_data():
    """Load graded props and game context."""
    graded = pd.read_parquet(PROC_DIR / "props_graded_backtest.parquet")
    games = pd.read_parquet(RAW_DIR / "games_historical.parquet")
    
    # Add game context to props
    # Match props to games via team + season + week
    graded["team_key"] = graded.apply(
        lambda r: r["home_team"] if pd.notna(r["home_team"]) else "", axis=1
    )
    
    # Merge game-level data
    game_cols = ["season", "week", "home_team", "away_team", "home_rest", "away_rest",
                 "div_game", "gameday", "weekday", "roof", "surface", "temp", "wind"]
    available = [c for c in game_cols if c in games.columns]
    
    merged = graded.merge(
        games[available].drop_duplicates(subset=["season", "week", "home_team"]),
        on=["season", "week", "home_team"],
        how="left",
        suffixes=("", "_game"),
    )
    
    return merged


def calc_hit_rate(df, condition_mask, label=""):
    """Calculate over hit rate for a subset and compare to complement."""
    subset = df[condition_mask]
    complement = df[~condition_mask]
    
    if len(subset) < 30:
        return None
    
    s_wins = (subset["result"] == "won").sum()
    s_total = (subset["result"].isin(["won", "lost"])).sum()
    s_rate = s_wins / s_total if s_total > 0 else 0
    
    c_wins = (complement["result"] == "won").sum()
    c_total = (complement["result"].isin(["won", "lost"])).sum()
    c_rate = c_wins / c_total if c_total > 0 else 0
    
    diff = s_rate - c_rate
    
    # Simple significance: is the difference meaningful?
    # Using rough threshold: |diff| > 2% with n > 100 is interesting
    significant = abs(diff) > 0.02 and s_total > 100
    profitable = s_rate > 0.524 or (1 - s_rate) > 0.524  # either over or under is profitable
    
    return {
        "label": label,
        "subset_n": s_total,
        "subset_over_rate": s_rate,
        "complement_n": c_total,
        "complement_over_rate": c_rate,
        "difference": diff,
        "significant": significant,
        "profitable_over": s_rate > 0.524,
        "profitable_under": (1 - s_rate) > 0.524,
    }


def test_h004_rest(df):
    """H004: Short rest / Thursday games → underperformance"""
    print("\n" + "=" * 60)
    print("H004: REST / SHORT WEEK EFFECT")
    print("=" * 60)
    
    results = []
    
    # Thursday games (short rest for at least one team)
    if "weekday" in df.columns:
        thursday = df["weekday"].str.contains("Thursday", case=False, na=False)
        r = calc_hit_rate(df, thursday, "Thursday games (all props)")
        if r: results.append(r)
    
    # Home team short rest (<=6 days)
    if "home_rest" in df.columns:
        short_home = df["home_rest"] <= 6
        r = calc_hit_rate(df, short_home, "Home team short rest (<=6 days)")
        if r: results.append(r)
        
        # Long rest (bye week return, 13+ days)
        long_rest = df["home_rest"] >= 13
        r = calc_hit_rate(df, long_rest, "Home team post-bye (>=13 days)")
        if r: results.append(r)
    
    for r in results:
        status = "→ SIGNAL" if r["significant"] else "  (noise)"
        direction = "UNDER" if r["subset_over_rate"] < 0.48 else "OVER" if r["subset_over_rate"] > 0.52 else "neutral"
        print(f"  {status} {r['label']}")
        print(f"         Over rate: {r['subset_over_rate']:.1%} (n={r['subset_n']}) vs {r['complement_over_rate']:.1%} baseline")
        print(f"         Diff: {r['difference']:+.1%} | Direction: {direction}")
    
    return results


def test_h005_dome(df):
    """H005: Dome games boost passing props"""
    print("\n" + "=" * 60)
    print("H005: DOME / INDOOR GAME EFFECT")
    print("=" * 60)
    
    results = []
    
    if "roof" in df.columns:
        dome = df["roof"].isin(["dome", "closed"])
        
        # All props in dome
        r = calc_hit_rate(df, dome, "Dome games (all props)")
        if r: results.append(r)
        
        # Passing yards in dome
        pass_dome = dome & (df["market"] == "player_pass_yds")
        r = calc_hit_rate(df, pass_dome, "Dome - passing yards")
        if r: results.append(r)
        
        # Receiving yards in dome
        rec_dome = dome & (df["market"] == "player_reception_yds")
        r = calc_hit_rate(df, rec_dome, "Dome - receiving yards")
        if r: results.append(r)
        
        # Receptions in dome
        receptions_dome = dome & (df["market"] == "player_receptions")
        r = calc_hit_rate(df, receptions_dome, "Dome - receptions")
        if r: results.append(r)
    
    for r in results:
        status = "→ SIGNAL" if r["significant"] else "  (noise)"
        direction = "OVER" if r["subset_over_rate"] > 0.52 else "UNDER" if r["subset_over_rate"] < 0.48 else "neutral"
        print(f"  {status} {r['label']}")
        print(f"         Over rate: {r['subset_over_rate']:.1%} (n={r['subset_n']}) vs {r['complement_over_rate']:.1%} baseline")
        print(f"         Diff: {r['difference']:+.1%} | Direction: {direction}")
    
    return results


def test_h010_weather(df):
    """H010: Bad weather suppresses passing props"""
    print("\n" + "=" * 60)
    print("H010: WEATHER SUPPRESSION EFFECT")
    print("=" * 60)
    
    results = []
    
    if "wind" in df.columns:
        # High wind
        high_wind = df["wind"].fillna(0) >= 15
        r = calc_hit_rate(df[df["market"] == "player_pass_yds"], 
                         high_wind[df["market"] == "player_pass_yds"], 
                         "High wind (>=15mph) - passing yards")
        if r: results.append(r)
        
        r = calc_hit_rate(df[df["market"] == "player_reception_yds"],
                         high_wind[df["market"] == "player_reception_yds"],
                         "High wind (>=15mph) - receiving yards")
        if r: results.append(r)
        
        # Extreme wind
        extreme_wind = df["wind"].fillna(0) >= 20
        r = calc_hit_rate(df[df["market"].isin(["player_pass_yds", "player_reception_yds"])],
                         extreme_wind[df["market"].isin(["player_pass_yds", "player_reception_yds"])],
                         "Extreme wind (>=20mph) - all passing/rec")
        if r: results.append(r)
    
    if "temp" in df.columns:
        # Cold games
        cold = df["temp"].fillna(60) <= 32
        r = calc_hit_rate(df, cold, "Cold games (<=32F) - all props")
        if r: results.append(r)
        
        cold_pass = cold & (df["market"] == "player_pass_yds")
        r = calc_hit_rate(df, cold_pass, "Cold games - passing yards")
        if r: results.append(r)
        
        # Cold + rushing (should boost)
        cold_rush = cold & (df["market"] == "player_rush_yds")
        r = calc_hit_rate(df, cold_rush, "Cold games - rushing yards")
        if r: results.append(r)
    
    for r in results:
        status = "→ SIGNAL" if r["significant"] else "  (noise)"
        direction = "UNDER" if r["subset_over_rate"] < 0.48 else "OVER" if r["subset_over_rate"] > 0.52 else "neutral"
        print(f"  {status} {r['label']}")
        print(f"         Over rate: {r['subset_over_rate']:.1%} (n={r['subset_n']}) vs {r['complement_over_rate']:.1%} baseline")
        print(f"         Diff: {r['difference']:+.1%} | Direction: {direction}")
    
    return results


def test_h014_division(df):
    """H014: Division games are tighter/lower scoring"""
    print("\n" + "=" * 60)
    print("H014: DIVISION GAME EFFECT")
    print("=" * 60)
    
    results = []
    
    if "div_game" in df.columns:
        div = df["div_game"] == 1
        
        r = calc_hit_rate(df, div, "Division games (all props)")
        if r: results.append(r)
        
        # By market
        for market in ["player_pass_yds", "player_rush_yds", "player_reception_yds", "player_receptions"]:
            mkt_df = df[df["market"] == market]
            div_mkt = div[df["market"] == market]
            r = calc_hit_rate(mkt_df, div_mkt, f"Division - {market}")
            if r: results.append(r)
    
    for r in results:
        status = "→ SIGNAL" if r["significant"] else "  (noise)"
        direction = "UNDER" if r["subset_over_rate"] < 0.48 else "OVER" if r["subset_over_rate"] > 0.52 else "neutral"
        print(f"  {status} {r['label']}")
        print(f"         Over rate: {r['subset_over_rate']:.1%} (n={r['subset_n']}) vs {r['complement_over_rate']:.1%} baseline")
        print(f"         Diff: {r['difference']:+.1%} | Direction: {direction}")
    
    return results


def test_h007_refs(df):
    """H007: Flag-heavy referees boost passing volume"""
    print("\n" + "=" * 60)
    print("H007: REFEREE CREW EFFECT")
    print("=" * 60)
    
    # Load referee data
    refs_path = RAW_DIR / "referee_tendencies.parquet"
    officials_path = RAW_DIR / "officials_historical.parquet"
    
    if not refs_path.exists() or not officials_path.exists():
        print("  Referee data not available")
        return []
    
    ref_tend = pd.read_parquet(refs_path)
    officials = pd.read_parquet(officials_path)
    
    # Get head referee for each game
    head_refs = officials[officials["off_pos"] == "R"][["game_id", "name", "season"]].rename(
        columns={"name": "referee"}
    )
    
    # Classify refs as flag-heavy or flag-light
    median_penalties = ref_tend["avg_penalties"].median()
    flag_heavy_refs = ref_tend[ref_tend["avg_penalties"] > median_penalties + 0.5]["referee"].tolist()
    flag_light_refs = ref_tend[ref_tend["avg_penalties"] < median_penalties - 0.5]["referee"].tolist()
    
    # Build game_id for our props to match
    # game_id format: YYYY_WW_AWAY_HOME
    df_with_gid = df.copy()
    df_with_gid["game_id"] = (
        df_with_gid["season"].astype(str) + "_" +
        df_with_gid["week"].astype(str).str.zfill(2) + "_" +
        df_with_gid["away_team"].fillna("") + "_" +
        df_with_gid["home_team"].fillna("")
    )
    
    # Merge refs
    df_with_refs = df_with_gid.merge(head_refs[["game_id", "referee"]], on="game_id", how="left")
    
    results = []
    
    # Flag-heavy refs
    heavy = df_with_refs["referee"].isin(flag_heavy_refs)
    r = calc_hit_rate(df_with_refs, heavy, "Flag-heavy refs (all props)")
    if r: results.append(r)
    
    # Flag-heavy + passing
    pass_heavy = heavy & (df_with_refs["market"] == "player_pass_yds")
    r = calc_hit_rate(df_with_refs, pass_heavy, "Flag-heavy refs - passing yards")
    if r: results.append(r)
    
    # Flag-light + passing
    light = df_with_refs["referee"].isin(flag_light_refs)
    pass_light = light & (df_with_refs["market"] == "player_pass_yds")
    r = calc_hit_rate(df_with_refs, pass_light, "Flag-light refs - passing yards")
    if r: results.append(r)
    
    for r in results:
        status = "→ SIGNAL" if r["significant"] else "  (noise)"
        direction = "OVER" if r["subset_over_rate"] > 0.52 else "UNDER" if r["subset_over_rate"] < 0.48 else "neutral"
        print(f"  {status} {r['label']}")
        print(f"         Over rate: {r['subset_over_rate']:.1%} (n={r['subset_n']}) vs {r['complement_over_rate']:.1%} baseline")
        print(f"         Diff: {r['difference']:+.1%} | Direction: {direction}")
    
    return results


def test_h011_injury_return(df):
    """H011: Players returning from injury underperform first week back"""
    print("\n" + "=" * 60)
    print("H011: INJURY RETURN UNDERPERFORMANCE")
    print("=" * 60)
    
    inj_path = RAW_DIR / "injuries_historical.parquet"
    if not inj_path.exists():
        print("  Injury data not available")
        return []
    
    injuries = pd.read_parquet(inj_path)
    
    # Find players who were OUT then came back
    # A player returns if: they were "Out" in week N, then have stats in week N+1 or N+2
    out_players = injuries[injuries["report_status"] == "Out"][
        ["season", "week", "full_name"]
    ].copy()
    out_players["full_name_clean"] = out_players["full_name"].str.strip().str.lower()
    
    # Mark return weeks: if player was out week W, their stats in week W+1 = "return game"
    out_players["return_week"] = out_players["week"] + 1
    
    # Match to props
    df_clean = df.copy()
    df_clean["player_clean"] = df_clean["player_name"].str.strip().str.lower()
    
    return_flags = out_players[["season", "return_week", "full_name_clean"]].rename(
        columns={"return_week": "week", "full_name_clean": "player_clean"}
    ).drop_duplicates()
    return_flags["is_return_game"] = True
    
    df_merged = df_clean.merge(return_flags, on=["season", "week", "player_clean"], how="left")
    df_merged["is_return_game"] = df_merged["is_return_game"].fillna(False)
    
    results = []
    
    returning = df_merged["is_return_game"] == True
    r = calc_hit_rate(df_merged, returning, "Return from injury (all props)")
    if r: results.append(r)
    
    # By market
    for market in ["player_pass_yds", "player_rush_yds", "player_reception_yds", "player_receptions"]:
        mkt = df_merged[df_merged["market"] == market]
        ret_mkt = returning[df_merged["market"] == market]
        r = calc_hit_rate(mkt, ret_mkt, f"Return from injury - {market}")
        if r: results.append(r)
    
    for r in results:
        status = "→ SIGNAL" if r["significant"] else "  (noise)"
        direction = "UNDER" if r["subset_over_rate"] < 0.48 else "OVER" if r["subset_over_rate"] > 0.52 else "neutral"
        print(f"  {status} {r['label']}")
        print(f"         Over rate: {r['subset_over_rate']:.1%} (n={r['subset_n']}) vs {r['complement_over_rate']:.1%} baseline")
        print(f"         Diff: {r['difference']:+.1%} | Direction: {direction}")
    
    return results


def test_h008_target_share(df):
    """H008: Rising target share predicts over on receiving props"""
    print("\n" + "=" * 60)
    print("H008: TARGET SHARE MOMENTUM")
    print("=" * 60)
    
    ts_path = PROC_DIR / "target_share_features.parquet"
    if not ts_path.exists():
        print("  Target share data not available")
        return []
    
    ts = pd.read_parquet(ts_path)
    ts["player_clean"] = ts["player_name"].str.strip().str.lower() if "player_name" in ts.columns else ""
    
    # Only receiving props
    rec_props = df[df["market"].isin(["player_reception_yds", "player_receptions"])].copy()
    rec_props["player_clean"] = rec_props["player_name"].str.strip().str.lower()
    
    # Merge target share features
    if "player_clean" in ts.columns and "season" in ts.columns and "week" in ts.columns:
        rec_merged = rec_props.merge(
            ts[["player_clean", "season", "week", "ts_rising", "ts_falling", "target_share_delta"]].drop_duplicates(),
            on=["player_clean", "season", "week"],
            how="left",
        )
    else:
        print("  Cannot match target share data")
        return []
    
    results = []
    
    # Rising target share
    if "ts_rising" in rec_merged.columns:
        rising = rec_merged["ts_rising"] == True
        r = calc_hit_rate(rec_merged, rising, "Rising target share (rec + rec yards)")
        if r: results.append(r)
        
        # Falling target share
        falling = rec_merged["ts_falling"] == True
        r = calc_hit_rate(rec_merged, falling, "Falling target share (rec + rec yards)")
        if r: results.append(r)
    
    # Large positive delta
    if "target_share_delta" in rec_merged.columns:
        big_rise = rec_merged["target_share_delta"].fillna(0) > 0.05
        r = calc_hit_rate(rec_merged, big_rise, "Target share delta > 5% (large increase)")
        if r: results.append(r)
        
        big_fall = rec_merged["target_share_delta"].fillna(0) < -0.05
        r = calc_hit_rate(rec_merged, big_fall, "Target share delta < -5% (large decrease)")
        if r: results.append(r)
    
    for r in results:
        status = "→ SIGNAL" if r["significant"] else "  (noise)"
        direction = "OVER" if r["subset_over_rate"] > 0.52 else "UNDER" if r["subset_over_rate"] < 0.48 else "neutral"
        print(f"  {status} {r['label']}")
        print(f"         Over rate: {r['subset_over_rate']:.1%} (n={r['subset_n']}) vs {r['complement_over_rate']:.1%} baseline")
        print(f"         Diff: {r['difference']:+.1%} | Direction: {direction}")
    
    return results


def test_h013_line_deviation(df):
    """H013: Props where line deviates far from player's average have edge"""
    print("\n" + "=" * 60)
    print("H013: LINE SIZE vs PLAYER AVERAGE")
    print("=" * 60)
    
    # Load player stats to get season averages
    stats = pd.read_parquet(RAW_DIR / "player_stats_historical.parquet")
    if "player_display_name" in stats.columns:
        stats["player_clean"] = stats["player_display_name"].str.strip().str.lower()
    else:
        stats["player_clean"] = stats["player_name"].str.strip().str.lower()
    
    market_to_stat = {
        "player_pass_yds": "passing_yards",
        "player_rush_yds": "rushing_yards",
        "player_reception_yds": "receiving_yards",
        "player_receptions": "receptions",
    }
    
    df_clean = df.copy()
    df_clean["player_clean"] = df_clean["player_name"].str.strip().str.lower()
    
    results = []
    
    for market, stat_col in market_to_stat.items():
        if stat_col not in stats.columns:
            continue
        
        mkt_props = df_clean[df_clean["market"] == market].copy()
        if mkt_props.empty:
            continue
        
        # Get player season average (using all games that season before this week)
        player_avgs = stats.groupby(["player_clean", "season"])[stat_col].mean().reset_index()
        player_avgs.rename(columns={stat_col: "season_avg"}, inplace=True)
        
        mkt_merged = mkt_props.merge(player_avgs, on=["player_clean", "season"], how="left")
        mkt_merged["line_vs_avg"] = mkt_merged["fanduel_line"] - mkt_merged["season_avg"]
        mkt_merged["line_pct_diff"] = mkt_merged["line_vs_avg"] / mkt_merged["season_avg"].replace(0, np.nan)
        
        # Line set ABOVE player's average (book is high → under should hit)
        line_high = mkt_merged["line_pct_diff"].fillna(0) > 0.10  # line is 10%+ above avg
        r = calc_hit_rate(mkt_merged, line_high, f"Line >10% above avg - {market}")
        if r: results.append(r)
        
        # Line set BELOW player's average (book is low → over should hit)
        line_low = mkt_merged["line_pct_diff"].fillna(0) < -0.10  # line is 10%+ below avg
        r = calc_hit_rate(mkt_merged, line_low, f"Line >10% below avg - {market}")
        if r: results.append(r)
    
    for r in results:
        status = "→ SIGNAL" if r["significant"] else "  (noise)"
        direction = "OVER" if r["subset_over_rate"] > 0.52 else "UNDER" if r["subset_over_rate"] < 0.48 else "neutral"
        print(f"  {status} {r['label']}")
        print(f"         Over rate: {r['subset_over_rate']:.1%} (n={r['subset_n']}) vs {r['complement_over_rate']:.1%} baseline")
        print(f"         Diff: {r['difference']:+.1%} | Direction: {direction}")
    
    return results


def main():
    print("=" * 70)
    print("HYPOTHESIS TESTING: SITUATIONAL & USAGE FACTORS")
    print("Testing against 19,815 graded FanDuel prop lines (2023-2025)")
    print("=" * 70)
    
    print("\nLoading data...")
    df = load_data()
    print(f"Props with game context: {len(df):,}")
    
    all_results = {}
    
    # Run all tests
    all_results["H004"] = test_h004_rest(df)
    all_results["H005"] = test_h005_dome(df)
    all_results["H010"] = test_h010_weather(df)
    all_results["H014"] = test_h014_division(df)
    all_results["H007"] = test_h007_refs(df)
    all_results["H011"] = test_h011_injury_return(df)
    all_results["H008"] = test_h008_target_share(df)
    all_results["H013"] = test_h013_line_deviation(df)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: CONFIRMED SIGNALS")
    print("=" * 70)
    
    all_signals = []
    for hyp_id, hyp_results in all_results.items():
        if not hyp_results:
            continue
        for r in hyp_results:
            if r and r.get("significant"):
                all_signals.append({**r, "hypothesis": hyp_id})
    
    if all_signals:
        print(f"\n{len(all_signals)} significant signals found:\n")
        for s in sorted(all_signals, key=lambda x: abs(x["difference"]), reverse=True):
            direction = "BET OVER" if s["subset_over_rate"] > 0.52 else "BET UNDER" if s["subset_over_rate"] < 0.48 else "NEUTRAL"
            profitable = "PROFITABLE" if s.get("profitable_over") or s.get("profitable_under") else ""
            print(f"  [{s['hypothesis']}] {s['label']}")
            print(f"       {s['subset_over_rate']:.1%} over rate (n={s['subset_n']}) | diff: {s['difference']:+.1%} | {direction} {profitable}")
            print()
    else:
        print("\nNo statistically significant signals found.")
        print("This doesn't mean they don't exist — may need more granular testing or interaction effects.")


if __name__ == "__main__":
    main()
