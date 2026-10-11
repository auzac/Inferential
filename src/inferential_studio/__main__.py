"""Allow ``python -m inferential_studio`` to launch the desktop application."""

from inferential_studio.app import main

if __name__ == "__main__":
    raise SystemExit(main())
