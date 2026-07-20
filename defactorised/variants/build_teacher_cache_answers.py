from _forward import launch

launch("build_teacher_cache.py", ("--generation-only",), ("--index-only", "--generation-only"))
