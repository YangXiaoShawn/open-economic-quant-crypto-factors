PY ?= python3
export PYTHONPATH := src:../packages/oeq-core/src

.PHONY: setup test reproduce-sample report dashboard check

setup:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

reproduce-sample:
	$(PY) -m crypto_factors.cli reproduce-sample

report:
	$(PY) src/crypto_factors/report_postpub.py && $(PY) src/crypto_factors/make_charts_postpub.py

dashboard:
	$(PY) src/crypto_factors/live_monitor.py

check: test reproduce-sample
