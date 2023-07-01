#!/usr/bin/env python3

import sys
import pathlib

from . import model
from . import cli

def main():
    _config, _positionals = cli.main(sys.argv[1:])

    if "version" in _config.keys():
        print("dalle_mini_terminal v1.1.0")
        sys.exit(0)
    elif "help" in _config.keys():
        print("dalle_mini_terminal --dalle dalle/artifacts/dir --vqgan vqgan/artifacts/dir -- avocado chair")
        sys.exit(0)

    dalle_dir = _config.get("dalle", "./dalle-artifacts")
    vqgan_dir = _config.get("vqgan", "./vqgan-artifacts")
    output_dir = _config.get("output", ".")

    prompt = ' '.join(_positionals)
    print("Generating images with prompt:", prompt)

    model.main(prompt, dalle_dir, vqgan_dir, output_dir)
    sys.exit(0)

if __name__ == "__main__":
    main()

