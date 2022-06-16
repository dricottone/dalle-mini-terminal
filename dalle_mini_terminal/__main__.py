#!/usr/bin/env python3

import sys

from . import model
from . import cli

def main():
    _config, _positionals = cli.main(sys.argv[1:])

    if "version" in _config.keys():
        print("dalle_mini_terminal v1.0.0")
        sys.exit(0)
    elif "help" in _config.keys():
        print("dalle_mini_terminal --artifacts path/to/artifacts -- avocado chair")
        sys.exit(0)

    artifacts_dir = _config.get("artifacts", "./artifacts")

    prompt = ' '.join(_positionals)
    print("Generating images with prompt:", prompt)

    model.main(prompt, artifacts_dir)
    sys.exit(0)

if __name__ == "__main__":
    main()

