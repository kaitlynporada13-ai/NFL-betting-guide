"""
WEEKLY RUNBOOK — one command to run the full weekly workflow in order.

Usage:
  python -m pipeline.run_week            # early-week: validate, project, build card, log CLV
  python -m pipeline.run_week --close    # near kickoff: capture closing lines + score CLV

Early-week sequence (run Tue/Wed when lines first post):
  1. validate_all      re-validate every edge OOS; flag any that decayed
  2. prop_projections  blended model -> projection + call + confidence for every prop
  3. generate_bet_card ranked playable card (HIGH/MEDIUM-HIGH/MEDIUM) from the projections
  4. capture_lines     snapshot current lines (this is our "bet" line for CLV)
  5. clv record        log the playable picks at the captured line + timestamp

Near-kickoff sequence (run Sunday AM with --close):
  6. capture_lines     snapshot the closing lines
  7. clv close         attach closing line to logged picks + compute CLV
  8. clv report        sharpness scorecard (avg CLV, % positive)
"""
import sys
import traceback


def _step(label, fn):
    print("\n" + "#" * 72)
    print(f"# {label}")
    print("#" * 72)
    try:
        fn()
        return True
    except Exception as e:
        print(f"[run_week] STEP FAILED: {label}\n{e}")
        traceback.print_exc()
        return False


def early_week():
    from pipeline import validate_all, prop_projections, generate_bet_card, capture_lines, clv_tracker
    _step("1/5 Re-validate all edges (OOS health check)", validate_all.main)
    _step("2/5 Build blended-model projections for every prop", prop_projections.main)
    _step("3/5 Generate ranked bet card", generate_bet_card.generate_bet_card)
    _step("4/5 Capture current lines (our bet line for CLV)", capture_lines.capture_all)
    _step("5/5 Log playable picks to CLV tracker", clv_tracker.record_picks)
    print("\n[run_week] Early-week run complete. Review the bet card, place bets, then run "
          "`python -m pipeline.run_week --close` near kickoff to score CLV.")


def close_out():
    from pipeline import capture_lines, clv_tracker
    _step("6/8 Capture closing lines", capture_lines.capture_all)
    _step("7/8 Attach closing lines + compute CLV", clv_tracker.update_closing)
    _step("8/8 CLV sharpness scorecard", clv_tracker.clv_report)
    print("\n[run_week] Close-out complete. Positive avg CLV = we're beating the number.")


if __name__ == "__main__":
    if "--close" in sys.argv:
        close_out()
    else:
        early_week()
