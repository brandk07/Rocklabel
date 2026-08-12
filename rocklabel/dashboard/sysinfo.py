"""Is the LiDAR alive, and does this machine have what training needs?

The sensor probe answers one question the CLI never could: *before* you launch a
capture, is the multiScan actually streaming? Two independent signals, because
either alone lies:

* **Reachable** — an ICMP ping to the sensor IP. Cheap, but a powered sensor
  that is not configured to stream still answers it.
* **Streaming** — bind the Compact UDP port and count datagrams for a moment.
  This is the signal that matters. It is skipped while a live job is running,
  because that job holds the port and stealing datagrams from it would put
  holes in the user's recording.

Everything here is cached with a short TTL and probed off the request thread, so
a dead sensor costs the dashboard a timeout once, not on every poll.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import threading
import time

#: Seconds a probe result stays fresh.
_TTL = 4.0
#: How long the UDP listen probe waits for datagrams.
_LISTEN_SEC = 0.6

_cache: dict[str, tuple[float, dict]] = {}
_lock = threading.Lock()


def _cached(key: str, fn, ttl: float = _TTL):
    now = time.time()
    with _lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < ttl:
            return hit[1]
    value = fn()
    with _lock:
        _cache[key] = (now, value)
    return value


# --------------------------------------------------------------------------- #
# sensor
# --------------------------------------------------------------------------- #
def _ping(host: str, timeout_s: float = 1.0) -> dict:
    if not shutil.which("ping"):
        return {"reachable": None, "rtt_ms": None, "detail": "ping not installed"}
    try:
        t0 = time.perf_counter()
        proc = subprocess.run(
            ["ping", "-n", "-c", "1", "-W", str(int(max(1, timeout_s))), host],
            capture_output=True, text=True, timeout=timeout_s + 2.0,
        )
        rtt = (time.perf_counter() - t0) * 1000.0
    except (subprocess.SubprocessError, OSError) as e:
        return {"reachable": None, "rtt_ms": None, "detail": str(e)}
    if proc.returncode == 0:
        return {"reachable": True, "rtt_ms": round(rtt, 1), "detail": "responds to ping"}
    return {"reachable": False, "rtt_ms": None,
            "detail": "no ICMP reply — check power, cabling and the host's subnet"}


def _listen(port: int, bind: str = "", seconds: float = _LISTEN_SEC) -> dict:
    """Count Compact datagrams on ``port``. Never steals from a live job."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        sock.settimeout(0.15)
        sock.bind((bind, port))
    except OSError as e:
        sock.close()
        return {"streaming": None, "packets": 0, "bytes": 0,
                "detail": f"cannot bind udp/{port}: {e}"}

    packets = nbytes = 0
    deadline = time.time() + seconds
    try:
        while time.time() < deadline:
            try:
                data, _addr = sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            packets += 1
            nbytes += len(data)
    finally:
        sock.close()

    if packets:
        rate = packets / seconds
        return {"streaming": True, "packets": packets, "bytes": nbytes,
                "packets_per_sec": round(rate),
                "detail": f"{rate:.0f} datagrams/s on udp/{port}"}
    return {"streaming": False, "packets": 0, "bytes": 0, "packets_per_sec": 0,
            "detail": f"silent on udp/{port} — sensor powered but not streaming, "
                      f"or streaming to a different port/host"}


def sensor_status(sensor_ip: str, udp_port: int, bind_host: str = "",
                  port_busy: bool = False) -> dict:
    """Combined sensor verdict for the Live page's status card."""

    def probe() -> dict:
        ping = _ping(sensor_ip)
        if port_busy:
            listen = {"streaming": None, "packets": 0, "packets_per_sec": 0,
                      "detail": "a live job owns this port — its own stats are "
                                "the authoritative reading"}
        else:
            listen = _listen(udp_port, bind_host)
        if listen.get("streaming"):
            state, label = "streaming", "Streaming"
        elif port_busy:
            state, label = "busy", "In use by a job"
        elif ping.get("reachable"):
            state, label = "idle", "Reachable, not streaming"
        elif ping.get("reachable") is False:
            state, label = "offline", "Offline"
        else:
            state, label = "unknown", "Unknown"
        return {
            "state": state, "label": label,
            "sensor_ip": sensor_ip, "udp_port": udp_port,
            "ping": ping, "listen": listen,
            "checked": time.time(),
        }

    return _cached(f"sensor:{sensor_ip}:{udp_port}:{port_busy}", probe)


def local_interfaces() -> list[dict]:
    """Local IPv4 addresses — the fast answer to "am I on the sensor's subnet?"."""

    def probe() -> list[dict]:
        out = []
        try:
            import psutil

            for name, addrs in psutil.net_if_addrs().items():
                for a in addrs:
                    if a.family == socket.AF_INET and not a.address.startswith("127."):
                        out.append({"name": name, "address": a.address,
                                    "netmask": a.netmask or ""})
        except Exception:
            try:
                out.append({"name": "default", "address": socket.gethostbyname(
                    socket.gethostname()), "netmask": ""})
            except OSError:
                pass
        return out

    return _cached("ifaces", probe, ttl=30.0)


def same_subnet(sensor_ip: str) -> bool | None:
    """Whether any local interface shares the sensor's /24. None if unknown."""
    prefix = sensor_ip.rsplit(".", 1)[0] if sensor_ip.count(".") == 3 else None
    if not prefix:
        return None
    ifaces = local_interfaces()
    if not ifaces:
        return None
    return any(i["address"].rsplit(".", 1)[0] == prefix for i in ifaces)


# --------------------------------------------------------------------------- #
# machine
# --------------------------------------------------------------------------- #
def _gpu() -> list[dict]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        proc = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4.0,
        )
    except (subprocess.SubprocessError, OSError):
        return []
    gpus = []
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            gpus.append({
                "name": parts[0],
                "memory_used_mb": int(float(parts[1])),
                "memory_total_mb": int(float(parts[2])),
                "utilization": int(float(parts[3])),
                "temperature_c": int(float(parts[4])),
            })
        except ValueError:
            continue
    return gpus


def machine(root: str) -> dict:
    """CPU / RAM / GPU / disk, for the header strip."""

    def probe() -> dict:
        out: dict = {"gpus": _gpu()}
        try:
            import psutil

            out["cpu_percent"] = psutil.cpu_percent(interval=None)
            out["cpu_count"] = psutil.cpu_count(logical=True)
            vm = psutil.virtual_memory()
            out["memory_used"] = vm.used
            out["memory_total"] = vm.total
            out["memory_percent"] = vm.percent
        except Exception:
            out["cpu_percent"] = None
        try:
            usage = shutil.disk_usage(root)
            out["disk_free"] = usage.free
            out["disk_total"] = usage.total
        except OSError:
            pass
        out["hostname"] = socket.gethostname()
        out["pid"] = os.getpid()
        return out

    return _cached("machine", probe, ttl=2.5)


def torch_status() -> dict:
    """Is the [train] extra importable and is CUDA usable?

    Importing torch costs a couple of seconds, so this is cached for the life of
    the process and only touched when the Models page asks for it.
    """

    def probe() -> dict:
        try:
            import torch
        except ImportError as e:
            return {"available": False, "detail": f"torch not installed ({e}); "
                                                  "pip install -e '.[train]'"}
        cuda = bool(torch.cuda.is_available())
        return {
            "available": True,
            "version": torch.__version__,
            "cuda": cuda,
            "device": (torch.cuda.get_device_name(0) if cuda else "cpu"),
            "detail": "CUDA ready" if cuda else "CPU only — training will be slow",
        }

    return _cached("torch", probe, ttl=3600.0)
