# NFL Prop Betting Discovery & Feature Engineering Spec

## Primary Objective

Discover hidden predictive relationships that sportsbooks, fantasy models, and conventional analytics may not fully capture. Every suggestion is a hypothesis to test — not assumed truth. Focus on relationships that generalize across seasons.

## Evaluation Criteria (Every Variable)

- Statistical significance
- Predictive power (does it reduce MAE or improve hit rate?)
- Stability across seasons (2023, 2024, 2025 separately)
- Position-specific relevance
- Market inefficiency (is it already priced in?)
- Interaction effects with other variables
- Survives out-of-sample testing

---

## Research Areas

### 1. Human & Physiological Factors
- Injury type/severity/weeks missed/recovery curves
- Snap count after return
- Career workload / touches / fatigue indicators
- Short week vs normal rest / bye week effects
- International/cross-country travel / time zone changes
- Weather: heat, cold, rain, snow, wind, humidity, altitude
- Surface: dome vs outdoor, grass vs turf
- Stadium-specific performance

### 2. Psychological / Motivation Factors
- Revenge games (former team/coach/coordinator)
- Contract year / newly signed / franchise tag / trade request
- Milestone/record-chasing games
- Prime-time / national TV games
- Previous-week criticism / fumble / drops / benching
- Returning to starting lineup
- Birthday proximity

### 3. Relationship Graphs
- QB → WR/TE/RB chemistry
- Coordinator → position group usage
- Target concentration / trust under pressure
- Red-zone trust / third-down trust / two-minute drill trust
- Historical chemistry between specific player pairs

### 4. Coaching Fingerprints
- Long-term tendencies by HC/OC/DC
- Personnel usage, motion rate, play action, tempo
- Red-zone tendencies, target concentration
- RB committee patterns, rookie vs veteran usage

### 5. Scheme Interactions
- Coverage schemes (man/zone/Cover 1-4/quarters)
- Blitz frequency and pressure packages
- Player archetype performance vs specific defensive structures
- Single-high vs two-high safety tendencies

### 6. Player Archetypes
- Cluster by physical traits + usage + route tree + target depth
- Compare against archetype peers, not positional averages
- Identify role: workhorse, committee, breakout candidate, decoy, etc.

### 7. Hidden Usage Metrics
- Routes run, snap %, route participation
- Air yards, expected fantasy points, target share
- Red-zone/inside-10/inside-5/goal-line opportunities
- Third-down/two-minute/garbage-time usage

### 8. Offensive Identity Detection
- Pass-heavy/run-heavy transitions
- Coordinator/QB/personnel changes
- Injury-driven identity changes
- Detect fundamental changes vs. noise

### 9. Defensive Adaptation
- Overreaction to previous opponents
- Recently exposed in specific areas → creates opportunities elsewhere

### 10. Hidden Team/Player States
- Team: improving/declining/confident/fatigued/aggressive/conservative
- Player: role expanding/shrinking, breakout candidate, trust increasing/decreasing, snap restriction, decoy

### 11. Opponent Micro Matchups
- WR vs CB, slot vs slot CB, TE vs LB, RB vs LB
- Shadow/bracket coverage, double teams, personnel packages

### 12. Officials
- Referee crew penalty tendencies
- Impact on passing volume, scoring, game pace
- Specific penalty types (DPI, holding, illegal contact)

### 13. Vegas Market Intelligence
- Opening vs closing line movement
- Reverse line movement / steam moves
- Public % vs sharp %
- Closing line value as predictive signal

### 14. Residual Analysis
- For each prop: expected vs actual → residual
- Cluster large residuals, identify recurring pre-game factors
- Discover hidden variables explaining systematic misses

### 15. Open Discovery
- Unexpected interaction effects
- Unknown latent variables
- Market blind spots
- Temporal/behavioral effects
- Novel patterns not listed above
