"""Re-pull player + team stats (all seasons) with the fixed column mapping."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.ingest_stats import pull_player_stats, pull_team_stats
from pipeline.config_loader import get_data_dir

RAW = get_data_dir("raw")
seasons = [2021, 2022, 2023, 2024, 2025]

print("Re-pulling player stats...")
ps = pull_player_stats(seasons)
ps.to_parquet(RAW / "player_stats_historical.parquet", index=False)
print(f"  Saved {len(ps)} player-week rows")

# Verify 2025 teams now populated
p25 = ps[ps["season"] == 2025]
print(f"  2025 rows: {len(p25)}, recent_team non-null: {p25['recent_team'].notna().sum()}")
print(f"  2025 teams: {sorted(p25['recent_team'].dropna().unique().tolist())[:10]}...")

print("\nRe-pulling team stats...")
ts = pull_team_stats(seasons)
ts.to_parquet(RAW / "team_stats_historical.parquet", index=False)
print(f"  Saved {len(ts)} team-week rows")
t25 = ts[ts["season"] == 2025]
print(f"  2025 team-week rows: {len(t25)}")
print("DONE")
