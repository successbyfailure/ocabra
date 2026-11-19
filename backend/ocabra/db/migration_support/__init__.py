"""Code that only Alembic migrations run.

Frozen on purpose: a migration that has not run yet on some database still
executes this code, so live modules must not import from here (and nothing
here should follow changes to live code it once copied).
"""
