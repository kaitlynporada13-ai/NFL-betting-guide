import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from prop_view import render_market

render_market("Pass Yds", "🎯")
