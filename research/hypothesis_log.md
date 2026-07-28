# Hypothesis Log

Track every hypothesis tested, results, and whether it's incorporated into the model.

**Status codes:**
- `TESTING` — Currently being evaluated
- `CONFIRMED` — Statistically significant, incorporated into model
- `WEAK` — Some signal but not strong enough alone (may use in combination)
- `REJECTED` — No predictive value found, do not revisit
- `NEEDS_DATA` — Cannot test yet, requires additional data source
- `QUEUED` — Not yet tested

---

## Format

### [Hypothesis ID] — Short description
- **Area:** (from discovery spec)
- **Hypothesis:** What we expect to find
- **Data used:** What data was tested against
- **Method:** How it was tested
- **Result:** Statistical findings
- **Hit rate impact:** Did it improve prop prediction hit rate?
- **Status:** CONFIRMED / WEAK / REJECTED / NEEDS_DATA / QUEUED
- **Date tested:** 
- **Notes:**

---

## Tested Hypotheses

### H001 — Baseline FanDuel Over rate
- **Area:** Baseline
- **Hypothesis:** FanDuel overs hit ~50% (efficient market)
- **Data used:** 19,815 graded props (2023-2025)
- **Result:** Overs hit 48.4% overall. Market slightly favors unders.
- **Breakdown:** Pass yds 50.3%, Rec yds 49.4%, Receptions 47.3%, Rush yds 47.5%
- **Status:** CONFIRMED (market is efficient but slightly shaded toward overs)
- **Date tested:** 2026-07-28
- **Notes:** FanDuel sets lines slightly high on receptions and rushing to exploit public over bias. Passing yards is closest to fair. Plus-money overs are traps (40.6% hit rate).

### H002 — Blind over betting is unprofitable
- **Area:** Baseline
- **Hypothesis:** Blindly betting all overs loses money
- **Result:** -7.6% ROI. Confirms juice is working as intended.
- **Status:** CONFIRMED
- **Date tested:** 2026-07-28
- **Notes:** Need selective edge detection to be profitable. Break-even requires >52.4% hit rate at -110.

---

## Queued Hypotheses (Priority Order)

### H003 — Model prediction vs line direction
- **Area:** Core model validation
- **Hypothesis:** When our model predicts a stat significantly above FanDuel's line, the over hits at a profitable rate
- **Status:** QUEUED (next to test)

### H004 — Rest differential impact on props
- **Area:** Physiological
- **Hypothesis:** Players on short rest (Thu games, short weeks) underperform their props
- **Status:** QUEUED

### H005 — Dome game effect on passing props
- **Area:** Environmental
- **Hypothesis:** QBs playing in domes consistently exceed passing yard props
- **Status:** QUEUED

### H006 — Revenge game effect
- **Area:** Psychological
- **Hypothesis:** Players facing former teams outperform their props
- **Status:** QUEUED

### H007 — Referee crew impact on totals/props
- **Area:** Officials
- **Hypothesis:** Flag-heavy refs correlate with higher scoring and more passing volume
- **Status:** QUEUED

### H008 — Target share delta as leading indicator
- **Area:** Hidden usage
- **Hypothesis:** Players with rising target share (last 3 vs last 5) exceed receiving props
- **Status:** QUEUED

### H009 — Red zone opportunity rate
- **Area:** Hidden usage
- **Hypothesis:** Teams with high red zone trip rates produce more anytime TD winners
- **Status:** QUEUED

### H010 — Weather suppression on passing props
- **Area:** Environmental
- **Hypothesis:** Wind >15mph and rain significantly suppress passing yard props
- **Status:** QUEUED

### H011 — Injury return underperformance
- **Area:** Physiological
- **Hypothesis:** Players returning from 2+ week injuries underperform week 1 back
- **Status:** QUEUED

### H012 — QB-receiver chemistry (new teammates)
- **Area:** Relationship graphs
- **Hypothesis:** New WR/TE acquisitions underperform receiving props first 3 weeks with new QB
- **Status:** QUEUED

### H013 — Line size and over rate
- **Area:** Market intelligence
- **Hypothesis:** Props with lines far from a player's season average (high or low) have edge
- **Status:** QUEUED

### H014 — Division game scoring depression
- **Area:** Game context
- **Hypothesis:** Division games produce lower scoring (familiarity breeds tighter games), unders hit more
- **Status:** QUEUED

### H015 — Contract year performance
- **Area:** Psychological
- **Hypothesis:** Players in contract years consistently outperform early-season props
- **Status:** QUEUED

---

## Rejected Hypotheses

(None yet — testing hasn't started)

---

## Data Gaps (Needed for future hypotheses)

| Hypothesis | Missing Data | Potential Source |
|-----------|-------------|-----------------|
| Scheme interactions (H005+) | Coverage type per play | PFF ($40/mo) |
| WR vs CB matchups | Shadow coverage assignments | PFF |
| Snap counts | Per-play snap data | PFF |
| Contract year | Player contract status | Spotrac (free, manual) |
| Public betting % | Consensus pick data | Paid services |
| Line movement | Opening vs closing lines | The Odds API (have it) |
