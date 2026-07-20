from _forward import launch

launch("evaluate.py", ("--base",), ("--base", "--checkpoint", "--layer-residuals"))
