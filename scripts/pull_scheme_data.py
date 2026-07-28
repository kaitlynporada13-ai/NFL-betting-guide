import nfl_data_py as nfl
import pandas as pd

# The defense_man_zone_type column was only added for 2024+ season
# Pull without column filter and check what's available
print('Pulling 2024 PBP (full)...')
pbp = nfl.import_pbp_data([2024])
print(f'Columns: {len(pbp.columns)}')

scheme_cols = [c for c in pbp.columns if any(x in c.lower() for x in 
    ['coverage','man_zone','personnel','formation','motion'])]
print(f'Scheme columns found: {scheme_cols}')

if 'defense_man_zone_type' in pbp.columns:
    plays = pbp[pbp['play_type'].isin(['pass','run'])].copy()
    print(f'Man/Zone data: {plays["defense_man_zone_type"].notna().sum()} of {len(plays)}')
    print(plays['defense_man_zone_type'].value_counts())
else:
    # Try the columns that DO exist
    plays = pbp[pbp['play_type'].isin(['pass','run'])].copy()
    # Check for offense_personnel and formation
    for c in ['offense_personnel','defense_personnel','offense_formation']:
        if c in plays.columns:
            print(f'{c}: {plays[c].notna().sum()} non-null')
            print(plays[c].value_counts().head(5))
            print()
    
    # Build scheme features from what we have
    # Shotgun rate, no-huddle rate, pass rate by team
    team_scheme = plays.groupby(['season','defteam']).agg(
        total_plays=('play_type','count'),
        pass_plays=('pass_attempt','sum'),
        shotgun_plays=('shotgun','sum'),
        no_huddle_plays=('no_huddle','sum'),
    ).reset_index()
    team_scheme['pass_rate'] = team_scheme['pass_plays'] / team_scheme['total_plays']
    team_scheme['shotgun_rate'] = team_scheme['shotgun_plays'] / team_scheme['total_plays']
    team_scheme['no_huddle_rate'] = team_scheme['no_huddle_plays'] / team_scheme['total_plays']
    
    team_scheme.to_parquet('data/raw/team_scheme_2024.parquet', index=False)
    print('Saved team scheme data')
    print(team_scheme.sort_values('pass_rate', ascending=False).head(10))
