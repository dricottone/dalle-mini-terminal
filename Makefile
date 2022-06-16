#PYTHON_BIN=python3
PYTHON_BIN=python

#PIP_BIN=$(PYTHON_BIN) -m pip
PIP_BIN=pip

# NOTE: `pipx` not currently used
#PIPX_BIN=$(PYTHON_BIN) -m pipx
PIPX_BIN=pipx

VENV_BIN=$(PYTHON_BIN) -m venv

PY_COMPILE_BIN=$(PYTHON_BIN) -m py_compile

# NOTE: `pyproject-build` not currently used
#PYPROJECT_BUILD_BIN=$(PYTHON_BIN) -m build
PYPROJECT_BUILD_BIN=pyproject-build

# NOTE: `unittest` not currently used
#UNITTEST_BIN=$(PYTHON_BIN) -m unittest
UNITTEST_BIN=unittest --color

# NOTE: `mypy` not currently used
#MYPY_BIN=$(PYTHON_BIN) -m mypy
MYPY_BIN=MYPY_CACHE_DIR=dalle_mini_terminal/__mypycache__ mypy

# see https://git.dominic-ricottone.com/gap.git/about
#GAP_BIN=$(PYTHON_BIN) -m gap
GAP_BIN=gap

.PHONY: clean test install install-cuda uninstall run

clean:
	rm -rf **/__pycache__ **/*.pyc
	#rm -rf **/__mypycache__
	#rm -rf build
	#rm -rf *.egg-info

test:
	$(PY_COMPILE_BIN) dalle_mini_terminal/*.py
	#$(UNITTEST_BIN) --working-directory . tests --verbose
	#$(MYPY_BIN) -p dalle_mini_terminal

install:
	$(VENV_BIN) .venv
	(source .venv/bin/activate; $(PIP_BIN) install jax)
	(source .venv/bin/activate; $(PIP_BIN) install git+https://github.com/patil-suraj/vqgan-jax.git)
	(source .venv/bin/activate; $(PIP_BIN) install dalle-mini)
	$(GAP_BIN) dalle_mini_terminal/cli.toml -o dalle_mini_terminal/cli.py

install-cuda:
	$(VENV_BIN) .venv
	(source .venv/bin/activate; $(PIP_BIN) install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html)
	(source .venv/bin/activate; $(PIP_BIN) install git+https://github.com/patil-suraj/vqgan-jax.git)
	(source .venv/bin/activate; $(PIP_BIN) install dalle-mini)
	$(GAP_BIN) dalle_mini_terminal/cli.toml -o dalle_mini_terminal/cli.py

uninstall:
	rm -rf .venv

run:
	(source .venv/bin/activate; $(PYTHON_BIN) -m dalle_mini_terminal --artifacts ./mini-1_v0_artifacts -- cats playing chess)

