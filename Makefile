clean:
	rm -rf .venv
	rm -rf **/__pycache__ **/*.pyc

dalle_mini_terminal/cli.py: dalle_mini_terminal/cli.toml
	gap dalle_mini_terminal/cli.toml -o dalle_mini_terminal/cli.py

.venv: dalle_mini_terminal/cli.py
	python -m venv .venv
	(. .venv/bin/activate; pip install --upgrade pip)
	(. .venv/bin/activate; pip install wheel)
	(. .venv/bin/activate; pip install jax==0.3.25 jaxlib==0.3.25 orbax-checkpoint==0.1.1 git+https://github.com/patil-suraj/vqgan-jax.git dalle-mini)

install: .venv

run:
	(. .venv/bin/activate; python -m dalle_mini_terminal --dalle ./mini-1_v0_artifacts --vqgan ./vqgan_imagenet_f16_16384_artifacts -- cats playing chess)

build:
	sudo docker build -t dalle_mini_terminal .

.PHONY: clean install run build
