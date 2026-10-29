"""LAN entry point: python -m pyrobo.apps.web_sim --host 0.0.0.0 --port 8000."""
import argparse
from pathlib import Path
import socket


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--scene', type=Path, help='start each browser with this JSON case')
    parser.add_argument('--scenes-dir', type=Path, help='shared case library directory')
    parser.add_argument('--max-clients', type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not 1 <= args.max_clients <= 32:
        parser.error('port must be 1–65535, max-clients must be 1–32')
    import uvicorn
    from .scene import Scene
    from .server import create_app
    scene = Scene.model_validate_json(args.scene.read_bytes()) if args.scene else None
    app = create_app(scenes_dir=args.scenes_dir, initial_scene=scene, max_clients=args.max_clients)
    print(f'Local: http://127.0.0.1:{args.port}', flush=True)
    if args.host == '0.0.0.0':
        try:
            addresses = {entry[4][0] for entry in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)}
        except socket.gaierror:
            addresses = set()
        # UDP connect only selects a local route; no packet is sent.
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                probe.connect(('192.0.2.1', 9))
                addresses.add(probe.getsockname()[0])
        except OSError:
            pass
        for address in sorted(addresses):
            if not address.startswith('127.'):
                print(f'LAN:   http://{address}:{args.port}', flush=True)
    # One ASGI process owns workers and the shared scene store.
    uvicorn.run(app, host=args.host, port=args.port, ws_max_size=2_000_000)


if __name__ == '__main__':
    main()
