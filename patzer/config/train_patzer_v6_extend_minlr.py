# Continue v6 after its auto-cooldown finished.
# This keeps the same out_dir and resumes optimizer/model from ckpt.pt.
# LR schedule will effectively be at min_lr after cooldown completion; we just extend max_iters.

out_dir = 'checkpoints/patzer_v6'
init_from = 'resume'

# Extend training beyond the cooldown-determined stop.
# v6 ended ~217k; last ckpt was at 200k. Push this comfortably past that.
max_iters = 400000

# Keep eval cadence consistent with v6 so curves compare cleanly.
eval_interval = 1000
eval_iters = 50
log_interval = 100

# Keep early-stop behavior as in v6 (and auto_cooldown can no-op if cooldown already happened).
early_stop_patience_evals = 25
early_stop_min_iters = 150000
auto_cooldown = True

