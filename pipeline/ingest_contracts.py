"""
Contract Year Ingestion Module.
Loads contract year player data and builds features for the strategy engine.

Contract year hypothesis: Players in the final year of their deal have extra
motivation to perform, creating a slight OVER lean on their props.

Data source: data/human_notes/contract_year_2026.yaml (curated from Spotrac)
"""

import pandas as pd
import yaml
from pathlib import Path
from pipeline.config_loader import get_data_dir, get_project_root


def load_contract_year_players(season: int = 2026) -> pd.DataFrame:
    """
    Load all contract year players from the curated YAML file.
    Returns a DataFrame with columns:
        player, team, position, tier, notes, season
    """
    notes_dir = get_project_root() / "data" / "human_notes"
    contract_path = notes_dir / f"contract_year_{season}.yaml"

    if not contract_path.exists():
        print(f"  WARNING: No contract year file found for {season}")
        return pd.DataFrame(columns=["player", "team", "position", "tier", "notes", "season"])

    with open(contract_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    records = []

    # Tier 1 - high profile
    for p in data.get("tier_1_contract_year", []):
        records.append({
            "player": p["player"],
            "team": p.get("team", ""),
            "position": p.get("position", ""),
            "tier": 1,
            "age": p.get("age"),
            "notes": p.get("notes", ""),
            "season": season,
        })

    # Tier 2 - solid starters
    for p in data.get("tier_2_contract_year", []):
        records.append({
            "player": p["player"],
            "team": p.get("team", ""),
            "position": p.get("position", ""),
            "tier": 2,
            "age": p.get("age"),
            "notes": p.get("notes", ""),
            "season": season,
        })

    # Tier 3 - depth/rotational
    for p in data.get("tier_3_contract_year", []):
        records.append({
            "player": p["player"],
            "team": p.get("team", ""),
            "position": p.get("position", ""),
            "tier": 3,
            "age": p.get("age"),
            "notes": p.get("notes", ""),
            "season": season,
        })

    # QBs (lower priority)
    for p in data.get("qb_contract_year", []):
        records.append({
            "player": p["player"],
            "team": p.get("team", ""),
            "position": "QB",
            "tier": 4,  # lowest priority for betting
            "age": p.get("age"),
            "notes": p.get("notes", ""),
            "season": season,
        })

    df = pd.DataFrame(records)
    return df


def build_contract_year_features(season: int = 2026) -> pd.DataFrame:
    """
    Build a processed feature table for contract year players.
    Includes:
        - is_contract_year: bool
        - contract_tier: 1-4 (1 = highest motivation signal)
        - contract_boost: float multiplier for confidence adjustment
    """
    players = load_contract_year_players(season)

    if players.empty:
        return pd.DataFrame(columns=[
            "player", "team", "position", "is_contract_year",
            "contract_tier", "contract_boost", "season"
        ])

    # Ensure string columns are consistently typed
    for col in ["player", "team", "position", "notes"]:
        players[col] = players[col].fillna("").astype(str)

    # Assign confidence boost based on tier
    # Tier 1: Strong signal (+0.10 confidence boost)
    # Tier 2: Moderate signal (+0.07)
    # Tier 3: Weak signal (+0.04)
    # Tier 4 (QBs): Minimal (+0.02)
    tier_boost_map = {1: 0.10, 2: 0.07, 3: 0.04, 4: 0.02}

    players["is_contract_year"] = True
    players["contract_boost"] = players["tier"].map(tier_boost_map)

    # Rename for clarity
    players = players.rename(columns={"tier": "contract_tier"})

    # Save to processed
    output_path = get_data_dir("processed") / "contract_year_players.parquet"
    players.to_parquet(output_path, index=False)
    print(f"  Contract year players saved: {len(players)} players ({output_path.name})")

    return players


def is_contract_year_player(player_name: str, season: int = 2026) -> dict:
    """
    Quick lookup: Is this player in a contract year?
    Returns dict with contract_year info or empty dict if not found.
    
    Used by strategy_engine to adjust confidence on OVER bets.
    """
    proc_path = get_data_dir("processed") / "contract_year_players.parquet"

    if not proc_path.exists():
        # Build it if not cached
        build_contract_year_features(season)

    if not proc_path.exists():
        return {}

    df = pd.read_parquet(proc_path)

    # Fuzzy match on player name (handle "A.J. Brown" vs "AJ Brown" etc.)
    name_lower = player_name.lower().replace(".", "").replace("'", "")
    df["name_clean"] = df["player"].str.lower().str.replace(".", "", regex=False).str.replace("'", "", regex=False)

    match = df[df["name_clean"] == name_lower]

    if match.empty:
        # Try partial match
        match = df[df["name_clean"].str.contains(name_lower.split()[-1], na=False)]
        if len(match) > 1:
            # Multiple matches on last name, try first name too
            first_name = name_lower.split()[0] if " " in name_lower else ""
            if first_name:
                match = match[match["name_clean"].str.startswith(first_name)]

    if match.empty:
        return {}

    row = match.iloc[0]
    return {
        "is_contract_year": True,
        "contract_tier": int(row["contract_tier"]),
        "contract_boost": float(row["contract_boost"]),
        "team": row["team"],
        "position": row["position"],
        "notes": row.get("notes", ""),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("BUILDING CONTRACT YEAR FEATURES")
    print("=" * 60)

    df = build_contract_year_features(2026)
    print(f"\nTotal contract year players: {len(df)}")
    print(f"\nBy tier:")
    print(df.groupby("contract_tier")["player"].count().to_string())
    print(f"\nBy position:")
    print(df.groupby("position")["player"].count().to_string())

    # Test lookup
    print("\n\nTest lookups:")
    for name in ["Ja'Marr Chase", "CeeDee Lamb", "Patrick Mahomes", "Dalton Kincaid"]:
        result = is_contract_year_player(name)
        if result:
            print(f"  {name}: Tier {result['contract_tier']}, boost +{result['contract_boost']:.2f}")
        else:
            print(f"  {name}: NOT in contract year")
