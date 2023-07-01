FROM python:3.11.4-slim-bookworm

RUN apt-get update && apt-get install -y dumb-init git && apt-get clean
RUN pip install wheel
RUN pip install jax==0.3.25 jaxlib==0.3.25 orbax-checkpoint==0.1.1 git+https://github.com/patil-suraj/vqgan-jax.git dalle-mini

COPY dalle_mini_terminal /app/dalle_mini_terminal
VOLUME /app/dalle-artifacts
VOLUME /app/vqgan-artifacts
VOLUME /app/output

RUN python -c "exec('from huggingface_hub import hf_hub_download\nhf_hub_download(\"dalle-mini/dalle-mini\", filename=\"enwiki-words-frequency.txt\")')"

WORKDIR /app
ENTRYPOINT ["dumb-init", "--", "python", "-m", "dalle_mini_terminal", "--output", "./output", "--"]
CMD ["cats", "playing", "chess"]

