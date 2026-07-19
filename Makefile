# Loblaw Bio cell-count analysis
#
#   make setup      install dependencies
#   make pipeline   all of the analysis + fun stuff
#   make dashboard  start the dashboard
#
# Uses whichever Python is on PATH (GitHub Codespaces provides `python`).
PYTHON ?= python
PORT ?= 8501

.PHONY: setup pipeline dashboard clean

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

pipeline:
	$(PYTHON) load_data.py
	$(PYTHON) analysis.py

# headless + 0.0.0.0 so it runs non-interactively and Codespaces can forward
# the port (no first-run e-mail prompt, reachable through the proxy).
dashboard:
	$(PYTHON) -m streamlit run dashboard.py \
		--server.headless true \
		--server.address 0.0.0.0 \
		--server.port $(PORT)

clean:
	rm -f cell-count.db
	rm -f responders_vs_nonresponders_boxplot.png
	rm -f responders_vs_nonresponders_longitudinal.png
	rm -rf outputs __pycache__
