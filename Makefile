.PHONY: cmf clean environment linux_requirements python_requirements requirements precommit test

#################################################################################
# GLOBALS																	   #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = company_matching
SENSITIVE_PROJECT = no
PYTHON_VERSION = 3.9
PYTHON_INTERPRETER = python

NOW:=$(shell date +"%m-%d-%y_%H-%M-%S")

#################################################################################
# COMMANDS																	  #
#################################################################################

## Make datasets table
cmf:
	$(PYTHON_INTERPRETER) cmf/admin.py


## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Set up python interpreter environment
environment:
	conda env remove --name $(PROJECT_NAME)
ifneq ("$(wildcard conda.lock.yml)","")
	@echo ">>> Creating conda environment from conda lock file"
	conda env create -f conda.lock.yml
else
	@echo ">>> Creating conda environment from scratch"
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) pip setuptools wheel pip-tools
	conda env export --name $(PROJECT_NAME) > conda.lock.yml 
endif
	@echo ">>> New conda env created. Activate with:\nconda activate $(PROJECT_NAME)"


## Reformat, lint, clear notebook outputs if necessary
precommit:
	ruff format .
	ruff check . --fix
ifeq (yes,$(SENSITIVE_PROJECT))
	@echo "Clearing output of all notebooks:"
	export JUPYTER_CONFIG_DIR=${HOME}/.jupyter_conf; jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace notebooks/*.ipynb
endif
	@echo "Done."


## Install Linux dependencies
linux_requirements:
	sudo dw-install libpq-dev postgresql


## Install Python dependencies
python_requirements:
	$(PYTHON_INTERPRETER) -m piptools compile --extra dev --resolver=backtracking --output-file=requirements.txt pyproject.toml
	$(PYTHON_INTERPRETER) -m piptools sync requirements.txt
	$(PYTHON_INTERPRETER) -m pip install --editable .
	$(PYTHON_INTERPRETER) -m ipykernel install --user --name=$(PROJECT_NAME)
	pre-commit install


## Install combined dependencies
requirements:
	make linux_requirements
	make python_requirements


## Run Python tests
test:
	# pytest test/test_db.py test/test_cleaning.py test/test_helpers.py test/test_dedupers.py
	# pytest test/test_dedupers.py test/test_linkers.py test/test_utils.py 
	pytest test/test_dedupers.py
	# pytest test/test_helpers.py::test_single_table_with_model_query


#################################################################################
# Self Documenting Commands													 #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
