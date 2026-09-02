import os
from os.path import join
from setuptools import setup, Extension


# Build-time knobs, set as environment variables at install time:
#
#   BOSS_TOL_BIC=1e-2  minimum improvement (in BIC points) better_mutation will accept
#   CG_VERBOSE=1       print per-search sweep counts and best scores to stdout
#
# e.g.  BOSS_TOL_BIC=1e-1 CG_VERBOSE=1 pip install . --force-reinstall
extra_compile_args = ["-Wall", "-O3"]

if os.environ.get("CG_VERBOSE"):
    extra_compile_args.append("-DCG_VERBOSE")

_tol = os.environ.get("BOSS_TOL_BIC")
if _tol:
    extra_compile_args.append("-DBOSS_TOL_BIC=%s" % _tol)


ext_modules = [
  Extension(
    name="causalget.c_backend",
    sources=[join("causalget", "c_backend.c")],
    extra_compile_args=extra_compile_args,
  )
]

setup(
  name="causal-get",
  version="0.0.1",
  description="Causal Graph Estimation Toolbox",
  packages=["causalget"],
  ext_modules=ext_modules,
)
