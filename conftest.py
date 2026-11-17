"""Root conftest: puts the repo root on sys.path so `app`, `mlops`, and
`models` import as top-level packages under any pytest invocation
(the CI workflow runs the bare `pytest` console script, which does not
add the working directory to sys.path the way `python -m pytest` does).
"""
