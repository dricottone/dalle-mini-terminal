PYTHON_BIN=python3
PIP_BIN=$(PYTHON_BIN) -m pip
VENV_BIN=$(PYTHON_BIN) -m venv
PY_COMPILE_BIN=$(PYTHON_BIN) -m py_compile

# see https://git.dominic-ricottone.com/~dricottone/gap
GAP_BIN=gap

clean:
	rm -rf **/__pycache__ **/*.pyc

uninstall:
	rm -rf .venv

test:
	$(PY_COMPILE_BIN) dalle_mini_terminal/*.py

.venv:
	$(VENV_BIN) .venv

dalle_mini_terminal/cli.py:
	$(GAP_BIN) dalle_mini_terminal/cli.toml -o dalle_mini_terminal/cli.py

build: dalle_mini_terminal/cli.py

install: .venv dalle_mini_terminal/cli.py
	(source .venv/bin/activate; $(PIP_BIN) install jax)
	(source .venv/bin/activate; $(PIP_BIN) install git+https://github.com/patil-suraj/vqgan-jax.git)
	(source .venv/bin/activate; $(PIP_BIN) install dalle-mini)

install-cuda: .venv dalle_mini_terminal/cli.py
	(source .venv/bin/activate; $(PIP_BIN) install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html)
	(source .venv/bin/activate; $(PIP_BIN) install git+https://github.com/patil-suraj/vqgan-jax.git)
	(source .venv/bin/activate; $(PIP_BIN) install dalle-mini)

run:
	(source .venv/bin/activate; $(PYTHON_BIN) -m dalle_mini_terminal --artifacts ./mini-1_v0_artifacts -- cats playing chess)

.PHONY: clean uninstall test build install install-cuda run
