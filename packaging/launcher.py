"""PyInstaller entry script. Kept trivial: all logic lives in inferential_studio.app."""

from inferential_studio.app import main

if __name__ == "__main__":
    raise SystemExit(main())
