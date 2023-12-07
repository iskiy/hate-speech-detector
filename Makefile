# Makefile
SHELL = /bin/bash

# Styling
.PHONY: style
style:
	black .
	flake8 --exclude venv --ignore E501,F401,F403,F405
	python3 -m isort .
	pyupgrade
	ruff --ignore=F401,F405,F403,E501 .
# Cleaning
.PHONY: clean
clean: style
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	rm -rf .coverage*
