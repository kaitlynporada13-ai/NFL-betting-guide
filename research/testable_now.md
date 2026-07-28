# What We Can Test RIGHT NOW

Based on data we already have in the system.

## Tier 1 — Test Immediately (have all data)

| Hypothesis | Data Source |
|-----------|-------------|
| Model prediction vs line direction (H003) | Our model + 63K prop lines |
| Rest differential / short week (H004) | nflverse schedule (home_rest, away_rest) |
| Dome game passing boost (H005) | stadiums.yaml + prop lines |
| Referee crew impact (H007) | officials + penalties data |
| Target share delta (H008) | target_share_features.parquet |
| Red zone opportunity rate (H009) | redzone_features.parquet |
| Weather suppression (H010) | PBP has wind/temp + prop lines |
| Injury return underperformance (H011) | injuries_historical.parquet |
| Line size vs season average (H013) | prop lines + player rolling stats |
| Division game depression (H014) | schedule data (div_game) |

## Tier 2 — Can Build From Existing Data (needs feature engineering)

| Hypothesis | What's Needed |
|-----------|---------------|
| Revenge game (H006) | Cross-reference rosters year-over-year to find team changes |
| QB-receiver chemistry (H012) | Roster changes + early-season receiving stats |
| Offensive identity shifts (H008) | PBP play-type frequencies + rolling windows |
| Player archetype clustering | Combine data + usage patterns |
| Coaching fingerprints | PBP tendencies grouped by OC/HC |

## Tier 3 — Needs External Data

| Hypothesis | Missing | Source | Cost |
|-----------|---------|--------|------|
| Coverage scheme matchups | Man/zone snap data | PFF | $40/mo |
| WR vs CB shadow coverage | Assignment data | PFF | $40/mo |
| Snap counts per play | Participation data | PFF | $40/mo |
| Contract year status | Salary data | Spotrac | Free (manual) |
| Public betting % | Consensus data | Action Network | $30/mo |
| Opening lines (for CLV) | Already have via The Odds API | Query earlier timestamps | Already paid |

## Execution Order

1. H003 — Model vs line (THE key question: is our model better than FanDuel?)
2. H004 + H005 + H010 + H014 — Environmental/situational (quick wins, easy to test)
3. H008 + H009 — Usage-based features (target share + red zone)
4. H007 — Referee impact
5. H011 — Injury return patterns
6. H013 — Line size anomalies
7. H006 + H012 — Psychological/relationship (more complex feature engineering)
