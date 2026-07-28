import nfl_data_py as nfl
print('Pulling 2025 PBP...')
pbp = nfl.import_pbp_data([2025])
print(f'2025 columns: {len(pbp.columns)}')
scheme = [c for c in pbp.columns if 'man_zone' in c or 'coverage_type' in c or 'personnel' in c]
print(f'Scheme cols found: {scheme}')
if 'defense_man_zone_type' in pbp.columns:
    plays = pbp[pbp['play_type'].isin(['pass', 'run'])]
    tagged = plays['defense_man_zone_type'].notna().sum()
    print(f'Man/Zone tagged: {tagged} of {len(plays)} plays ({tagged/len(plays):.0%})')
    print(plays['defense_man_zone_type'].value_counts())
else:
    print('No man/zone column in 2025')
