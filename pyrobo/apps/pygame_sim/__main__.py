"""Launch with python -m pyrobo.apps.pygame_sim --help."""
import os
import sys

from .config import parse_args


def main(argv=None):
    args, config = parse_args(argv)
    os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
    if args.headless:
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        os.environ['SDL_AUDIODRIVER'] = 'dummy'
    try:
        from .app import App
    except ModuleNotFoundError as error:
        if error.name != 'pygame':
            raise
        print("pygame is optional. Install it with: python -m pip install -e '.[interactive]'", file=sys.stderr)
        return 2
    app = App(config, headless=args.headless)
    try:
        return app.run(steps=args.steps if args.steps is not None else (100 if args.headless else None),
                       screenshot=args.screenshot)
    finally:
        app.close()


if __name__ == '__main__':
    raise SystemExit(main())
