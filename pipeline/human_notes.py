"""
Human Notes System.
Loads qualitative insights from YAML files and integrates them into predictions.
Allows you to add context the model can't capture (revenge games, contract years, etc.)
"""

import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime

from pipeline.config_loader import get_data_dir, load_settings


def load_weekly_notes(week: int | None = None, season: int | None = None) -> dict | None:
    """
    Load human notes for the specified week.
    If no week specified, tries to find the most recent notes file.
    
    Returns:
        Dict with game_notes, player_notes, team_notes or None if no file found.
    """
    notes_dir = get_data_dir("human_notes")
    settings = load_settings()

    if season is None:
        season = settings["data"]["current_season"]

    if week is not None:
        # Look for specific week file
        filename = f"week_{week:02d}_notes.yaml"
        filepath = notes_dir / filename
        if filepath.exists():
            with open(filepath, "r") as f:
                return yaml.safe_load(f)
        return None

    # Find the most recent notes file
    note_files = sorted(notes_dir.glob("week_*_notes.yaml"), reverse=True)
    for filepath in note_files:
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)
            if data and data.get("season") == season:
                return data

    return None


def apply_notes_to_predictions(predictions: pd.DataFrame, notes: dict) -> pd.DataFrame:
    """
    Apply human notes to adjust predictions and add reasoning.
    
    Notes can:
    1. Adjust numerical predictions (e.g., +5 receiving yards)
    2. Adjust confidence levels
    3. Add narrative reasoning to display in dashboard
    """
    df = predictions.copy()

    if "reasoning" not in df.columns:
        df["reasoning"] = ""
    if "human_adjustment" not in df.columns:
        df["human_adjustment"] = 0.0

    # Apply player-level notes
    player_notes = notes.get("player_notes", [])
    for note in player_notes:
        player_name = note.get("player")
        note_text = note.get("notes", "")
        props_affected = note.get("props_affected", {})

        if "player_name" in df.columns:
            mask = df["player_name"].str.contains(player_name, case=False, na=False)
        elif "player_display_name" in df.columns:
            mask = df["player_display_name"].str.contains(player_name, case=False, na=False)
        else:
            continue

        if mask.any():
            # Add reasoning
            df.loc[mask, "reasoning"] = df.loc[mask, "reasoning"] + f" | HUMAN: {note_text.strip()}"

            # Apply adjustments to predictions
            for prop, adjustment in props_affected.items():
                if "prediction" in df.columns:
                    df.loc[mask, "prediction"] += adjustment
                    df.loc[mask, "human_adjustment"] += adjustment

    # Apply game-level notes
    game_notes = notes.get("game_notes", [])
    for note in game_notes:
        game_str = note.get("game", "")
        note_text = note.get("notes", "")
        confidence_adj = note.get("confidence_adjustment", 0)

        # Try to match game
        teams = game_str.replace(" vs ", ",").replace(" @ ", ",").split(",")
        if len(teams) >= 2:
            team1, team2 = teams[0].strip(), teams[1].strip()
            
            for col in ["home_team", "away_team"]:
                if col in df.columns:
                    mask = df[col].str.contains(team1, case=False, na=False) | \
                           df[col].str.contains(team2, case=False, na=False)
                    if mask.any():
                        df.loc[mask, "reasoning"] = (
                            df.loc[mask, "reasoning"] + f" | GAME NOTE: {note_text.strip()}"
                        )
                        break

    # Apply team-level notes
    team_notes = notes.get("team_notes", [])
    for note in team_notes:
        team = note.get("team", "")
        note_text = note.get("notes", "")

        for col in ["recent_team", "home_team", "away_team"]:
            if col in df.columns:
                mask = df[col] == team
                if mask.any():
                    df.loc[mask, "reasoning"] = (
                        df.loc[mask, "reasoning"] + f" | TEAM NOTE: {note_text.strip()}"
                    )

    return df


def create_week_notes(week: int, season: int | None = None):
    """Create a blank notes file for a specific week."""
    settings = load_settings()
    if season is None:
        season = settings["data"]["current_season"]

    notes_dir = get_data_dir("human_notes")
    filename = f"week_{week:02d}_notes.yaml"
    filepath = notes_dir / filename

    if filepath.exists():
        print(f"Notes file already exists: {filepath}")
        return filepath

    template = {
        "season": season,
        "week": week,
        "game_notes": [],
        "player_notes": [],
        "team_notes": [],
    }

    with open(filepath, "w") as f:
        yaml.dump(template, f, default_flow_style=False, sort_keys=False)

    print(f"Created blank notes file: {filepath}")
    print(f"Edit this file to add your qualitative insights before running predictions.")
    return filepath


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        week_num = int(sys.argv[1])
        create_week_notes(week_num)
    else:
        print("Usage: python -m pipeline.human_notes <week_number>")
        print("Creates a blank notes file for the specified week.")
