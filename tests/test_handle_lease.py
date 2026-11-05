"""Handle leases: hold_until is read, /renew extends it, lapsed handles go.

No docker and no GPU: State is built with a stub config, the app with a
pre-built state instead of its lifespan, and the client's renewer runs against
a tiny local HTTP server.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from broker import app as app_mod
from broker.state import State, candidates_to_stop_for_idle
from client import gpu_broker_client as gbc


def _state() -> State:
    return State(cfg=SimpleNamespace(roles={}), docker=None)


def test_handle_without_lease_never_expires():
    s = _state()
    h = s.create_handle("embed", "c", None)
    assert h.hold_until is None
    assert s.expire_handles(now=time.time() + 10**9) == []
    assert h.handle_id in s.handles


def test_lapsed_lease_is_dropped():
    s = _state()
    h = s.create_handle("embed", "dead-client", 30)
    assert s.expire_handles(now=time.time() + 10) == []
    gone = s.expire_handles(now=time.time() + 31)
    assert [g.handle_id for g in gone] == [h.handle_id]
    assert not s.role_has_active_holds("embed")


def test_lapsed_handle_no_longer_shields_idle_shutdown():
    s = _state()
    s.workers["embed"] = SimpleNamespace(
        idle_shutdown_at=time.time() - 1, last_request_at=0.0,
        cfg=SimpleNamespace(idle_shutdown_seconds=300))
    s.create_handle("embed", "dead-client", 1)
    assert candidates_to_stop_for_idle(s) == []          # shielded while held
    s.expire_handles(now=time.time() + 2)
    # The idle clock starts at expiry, like a /release: stoppable 300 s later.
    assert not s.role_has_active_holds("embed")
    assert s.workers["embed"].idle_shutdown_at == pytest.approx(time.time() + 300, abs=2)
    s.workers["embed"].idle_shutdown_at = time.time() - 1
    assert candidates_to_stop_for_idle(s) == ["embed"]


def test_renew_extends_by_own_lease_and_can_add_one():
    s = _state()
    h = s.create_handle("embed", "c", 30)
    first = h.hold_until
    time.sleep(0.01)
    assert s.renew_handle(h.handle_id).hold_until > first
    n = s.create_handle("embed", "c", None)
    assert s.renew_handle(n.handle_id, 60).hold_until == pytest.approx(time.time() + 60, abs=2)
    assert s.renew_handle("nope") is None


@pytest.fixture
def api():
    s = _state()
    app_mod.app.state.broker = s
    s.gpu_budgets = {}
    yield TestClient(app_mod.app), s   # not entered as a context: no lifespan, no docker


def test_renew_endpoint(api):
    client, s = api
    h = s.create_handle("embed", "c", 30)
    r = client.post("/renew", json={"handle": h.handle_id, "hold_seconds": 120})
    assert r.status_code == 200
    assert r.json()["hold_until"] == pytest.approx(time.time() + 120, abs=2)
    r = client.post("/renew", json={"handle": "missing"})
    assert r.status_code == 404 and "unknown handle" in r.text


def test_status_drops_lapsed_and_lists_holders(api, monkeypatch):
    client, s = api
    monkeypatch.setattr(app_mod.observability, "gpu_used_mb", lambda: 0)
    live = s.create_handle("embed", "ingest", 300)
    dead = s.create_handle("rerank", "crashed", 300)
    dead.hold_until = time.time() - 1
    body = client.get("/status").json()
    assert body["active_handles"] == 1
    assert [(h["handle"], h["client_id"]) for h in body["handles"]] == [(live.handle_id[:8], "ingest")]


class _FakeBroker(BaseHTTPRequestHandler):
    renews: list[dict] = []
    mode = "ok"   # ok | lost | old

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/acquire":
            out = {"handle": "h-1234567890", "endpoint": "http://x", "role": body["role"]}
            out["_hold"] = body.get("hold_seconds")
            _FakeBroker.acquired = body
            self._send(200, out)
        elif self.path == "/renew":
            _FakeBroker.renews.append(body)
            if self.mode == "lost":
                self._send(404, {"detail": "unknown handle"})
            elif self.mode == "old":
                self._send(404, {"detail": "Not Found"})
            else:
                self._send(200, {"handle": body["handle"], "hold_until": time.time() + 3})
        elif self.path == "/release":
            self._send(204, None)

    def _send(self, code, obj):
        data = b"" if obj is None else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture
def fake_broker():
    _FakeBroker.renews = []
    srv = HTTPServer(("127.0.0.1", 0), _FakeBroker)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


@pytest.mark.parametrize("mode,expect_max", [("ok", None), ("lost", 1), ("old", 1)])
def test_client_renews_while_held(fake_broker, mode, expect_max):
    _FakeBroker.mode = mode
    c = gbc.GpuBrokerClient(base_url=fake_broker, client_id="t")
    with c.acquire("embed", hold_seconds=3):   # renews every 1 s
        time.sleep(3.5)
    n = len(_FakeBroker.renews)
    if expect_max is None:
        assert n >= 3 and all(r["hold_seconds"] == 3 for r in _FakeBroker.renews)
    else:
        assert n == expect_max   # stops after the first 404, either kind


def test_client_default_lease_and_opt_out(fake_broker):
    _FakeBroker.mode = "ok"
    c = gbc.GpuBrokerClient(base_url=fake_broker, client_id="t")
    with c.acquire("embed"):
        pass
    assert _FakeBroker.acquired["hold_seconds"] == gbc.DEFAULT_LEASE_SECONDS
    with c.acquire("embed", hold_seconds=0):
        pass
    assert "hold_seconds" not in _FakeBroker.acquired
