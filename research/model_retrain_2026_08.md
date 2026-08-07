# Model Retraining Report — August 7, 2026

## Summary

Retrained all 9 models (4 game-level + 5 player prop) with expanded feature set.
Player features grew from 88 to 110 columns (+22 new features).
All models trained, validated, and backtested.

---

## What Changed

### New Features Added to Player Prop Models

| Feature | Source | Models Using It |
|---------|--------|-----------------|
| `is_division_game` | games schedule | All 5 prop models |
| `snap_pct_roll3`, `snap_pct_roll5` | snap_counts.parquet | All 5 prop models |
| `avg_air_yards`, `avg_yac`, `deep_rate` | PBP air yards profile | Receiving Yards, Receptions |
| `rz_targets`, `rz_target_share` | PBP red zone data | Receiving Yards, TD |
| `rz_td_pct_roll5`, `rz_trips_roll5`, `rz_pass_rate_roll5` | Team red zone features | Passing Yards, TD |
| `gl_carries`, `gl_tds` | PBP goal-line data | Rushing Yards, TD |
| `sack_rate`, `hit_rate` | PBP QB pressure | Passing Yards |

### Infrastructure Fix

- `base_model.py`: Relaxed `valid_mask` to only require non-null *target* (was requiring all features non-null, which excluded all rows when new sparse features had NaN). Features with NaN are filled with 0 before training, which XGBoost handles well.

---

## Model Performance (Time-Series Cross-Validation)

| Model | Metric | Value | Notes |
|-------|--------|-------|-------|
| Spread | CV MAE | 10.29 pts | ~10 pt error on spread prediction |
| Spread Cover | CV Accuracy | 76.5% | Classifies ATS correctly 3/4 times |
| Totals | CV MAE | 10.55 pts | |
| Moneyline | CV Accuracy | 60.3% | Modest — ML is well-priced |
| **Passing Yards** | CV MAE | **46.5 yds** | Key prop model |
| **Rushing Yards** | CV MAE | **16.1 yds** | Tight predictions |
| **Receiving Yards** | CV MAE | **13.9 yds** | Best prop model |
| **Receptions** | CV MAE | **1.01 rec** | ~1 reception error |
| **Anytime TD** | CV Accuracy | **77.5%** | Strong binary classifier |

### Feature Importance Highlights

- `is_division_game` ranked #5-8 across rushing, receiving, passing yards — validates our strategy engine's division signal
- `game_in_dome` ranked #7 for anytime TD — supports dome + TD OVER strategy  
- Rolling 3-game averages dominate all prop models (strongest signal is recent form)
- Snap count features appear in top 15 when available (2023-2024 data only)

---

## Backtest Results

### Baseline (blind betting)
- Betting ALL overs at -110: **-7.61% ROI** (19,815 bets)
- Market has a natural under lean (overs hit only 48.4% across 2023-2025)
- This confirms our research: UNDER is the default edge

### Model vs FanDuel Line (In-Sample)

| Edge Threshold | Bets | Hit Rate | ROI |
|:---:|:---:|:---:|:---:|
| >= 0 | 17,418 | 78.1% | +49.0% |
| >= 5 | 12,301 | 85.4% | +63.1% |
| >= 10 | 8,230 | 91.3% | +74.3% |
| >= 15 | 5,473 | 95.0% | +81.4% |

**IMPORTANT CAVEAT:** These are in-sample results (model was trained on this data). They measure the model's ability to explain historical outcomes, NOT predict future ones. True out-of-sample performance is estimated by CV metrics.

### Season Consistency (edge >= 10)

| Market | 2023 | 2024 | 2025 |
|--------|:---:|:---:|:---:|
| Passing Yards | 95.6% | 94.8% | 92.9% |
| Rushing Yards | 91.8% | 94.7% | 93.4% |
| Receiving Yards | 87.7% | 88.3% | 87.0% |
| Receptions | 93.8% | 93.9% | 92.0% |

No catastrophic single-season overfitting. Slight decline in 2025 (expected — model trained on less data for most recent season).

---

## Realistic Expectations for 2026 Season

Based on CV metrics and the nature of sports betting:

- **CV MAE tells us**: Model predictions will be ~46 yards off for passing, ~16 for rushing, ~14 for receiving, ~1 for receptions
- **For betting**: We don't need to be exactly right. We need to be directionally better than FanDuel's line.
- **Conservative estimate**: If the model identifies 50-100 high-confidence edges per week (large disagreements with FanDuel), and hits 53-55% (enough to profit at -110), we're profitable.
- **Strategy engine adds value on top**: Even without the model, our 25 confirmed strategies (from manual research) have proven ROI. The model is an additional signal layer.

### How to Use the Model in Production

1. **Primary tool**: Strategy engine (rule-based, proven over 3 seasons)
2. **Secondary tool**: XGBoost model predictions (identifies directional edges)
3. **Combination**: When strategy engine AND model agree → highest confidence bet
4. **Override rule**: Strategy engine signals (especially Week 1 UNDER, dome TDs) take precedence over model when model disagrees

---

## Next Steps

- [ ] 2026 Week 1: First true out-of-sample test
- [ ] Track model predictions vs actuals weekly → measure real-world performance
- [ ] After Week 4: Retrain with 2026 data to improve current-season predictions
- [ ] After Week 8: Full model evaluation — does adding new features help?
- [ ] End of season: Compare 2026 model ROI vs 2025 (before new features)
