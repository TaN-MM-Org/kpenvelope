"""The kpenvelope app: build a nitride layer stack in your web browser and
see its charges, its electron gas, its device curves and their error bars,
in plain words (new in v0.18.0).

Start it with ``kpenvelope-app`` or ``python -m kpenvelope.gui``. It runs on
your own computer (127.0.0.1) and needs nothing beyond kpenvelope and
NumPy.
"""
from .server import main, make_server

__all__ = ["main", "make_server"]
