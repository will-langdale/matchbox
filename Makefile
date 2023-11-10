.PHONY: check clear data dims validation setup evals train predict clean environment install_git_hooks requirements precommit test

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
datasets:
	$(PYTHON_INTERPRETER) cmf/data/datasets.py datasets


## Make dimension tables
dims:
	$(PYTHON_INTERPRETER) cmf/data/datasets.py dimensions


## Make probabilities table
probabilities:
	$(PYTHON_INTERPRETER) cmf/data/probabilities.py --overwrite


## Make clusters table with Companies House as the default data
clusters:
	$(PYTHON_INTERPRETER) cmf/data/clusters.py --dim_init companieshouse.companies --overwrite


## Make validation table
validation:
	$(PYTHON_INTERPRETER) cmf/data/validation.py --overwrite


## Setup system ready for linking
setup:
	make datasets
	make dims
	make probabilities
	make clusters
	make validation


## Make evaluation tables for existing matching service
evals:
	$(PYTHON_INTERPRETER) cmf/data/make_eval.py


## Shows disk usage across repo
check:
	du -hs ~/$(PROJECT_NAME)/.[^.]* * | sort -rh


## Removes all processed datasets
clear:
	rm -Rf data/processed/*


## Load data to parquet
data:
	$(PYTHON_INTERPRETER) cmf/data/get_datasets.py --output_dir company-matching__$(NOW) --sample 100000


## Train model
train:
	$(PYTHON_INTERPRETER) cmf/models/train.py --description "Initial test of the model training pipeline" --run_name company-matching__$(NOW) --input_dir company-matching__06-26-23_11-40-51 --dev


## Build lookup and write to data workspace
predict:
	$(PYTHON_INTERPRETER) cmf/models/predict.py --run --input_dir company-matching__06-26-23_11-40-51 --output_schema "_user_eaf4fd9a" --output_table "lge_lookup"


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


## Install pre-commit hook
install_git_hooks:
	pre-commit install

## Reformat, lint, clear notebook outputs if necessary
precommit:
	black . --extend-exclude \.ipynb$ 
	flake8 . --exclude scratch,.ipynb_checkpoints
ifeq (yes,$(SENSITIVE_PROJECT))
	@echo "Clearing output of all notebooks:"
	export JUPYTER_CONFIG_DIR=${HOME}/.jupyter_conf; jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace notebooks/*.ipynb
endif
	@echo "Done."

## Install Python Dependencies
requirements:
	$(PYTHON_INTERPRETER) -m piptools compile --output-file=requirements.txt --resolver=backtracking requirements.in requirements-dev.in
	$(PYTHON_INTERPRETER) -m piptools sync requirements.txt
	$(PYTHON_INTERPRETER) -m ipykernel install --user --name=$(PROJECT_NAME)
	make install_git_hooks

## Run Python tests
test:
	pytest test


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
