"""Why the wired link to the sensor is dead, and the commands that revive it.

When the multiScan is plugged straight into a laptop there is no DHCP server on
that cable, so NetworkManager keeps "helpfully" trying to configure the port,
fails, and tears the addresses back down. The port then looks up but carries no
usable IPv4, and the sensor's UDP stream lands nowhere. The fix is to take the
interface away from NetworkManager and set the three static addresses by hand:

* ``host_addr``    — this machine on the sensor's subnet, so SOPAS / the web UI
  is reachable and the sensor can ARP for us at all.
* ``udp_dest_addr`` — the address the sensor was told to stream Compact data to.
  It is on a *different* /24 than the sensor, which is exactly why it has to be
  added explicitly: nothing else would put that subnet on this NIC.
* ``gateway_addr``  — the sensor's configured gateway. The sensor routes its UDP
  packets via the gateway, so this host must answer to that address too.

This module never runs any of it. It reads the interface state, works out which
of those preconditions are unmet, and hands the browser a copy-pasteable script;
the commands need root, and silently running ``ip addr flush`` on someone's NIC
is not something a status card should do.

Linux-only, by nature — :func:`diagnose` reports ``supported: False`` elsewhere.
"""

from __future__ import annotations

import ipaddress
import os
import shutil
import socket
import subprocess
import sys
import threading
import time

#: Seconds an interface probe stays fresh (the Live page polls every 5 s).
_TTL = 4.0

_cache: dict[str, tuple[float, object]] = {}
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


def _run(argv: list[str], timeout: float = 3.0) -> str:
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (subprocess.SubprocessError, OSError):
        return ""
    return proc.stdout if proc.returncode == 0 else ""


# --------------------------------------------------------------------------- #
# interface inventory
# --------------------------------------------------------------------------- #
def _sysfs(iface: str, leaf: str) -> str:
    try:
        with open(f"/sys/class/net/{iface}/{leaf}", "r", encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return ""


def _is_physical_ethernet(iface: str) -> bool:
    """True for a real Ethernet NIC — not loopback, wifi, docker or veth.

    Virtual devices live under ``/sys/devices/virtual``, which is the one test
    that catches ``docker0``, bridges and veth pairs without a name blacklist.
    """
    if _sysfs(iface, "type") != "1":  # ARPHRD_ETHER
        return False
    if os.path.exists(f"/sys/class/net/{iface}/wireless") or \
            os.path.exists(f"/sys/class/net/{iface}/phy80211"):
        return False
    try:
        return "/devices/virtual/" not in os.path.realpath(f"/sys/class/net/{iface}")
    except OSError:
        return False


def _addrs_by_iface() -> dict[str, list[str]]:
    """``{iface: ["10.11.10.5/24", ...]}`` for every IPv4 currently assigned."""
    out: dict[str, list[str]] = {}
    if shutil.which("ip"):
        for line in _run(["ip", "-o", "-4", "addr", "show"]).splitlines():
            parts = line.split()
            # "2: enp0s31f6    inet 10.11.10.5/24 scope global enp0s31f6"
            if len(parts) >= 4 and parts[2] == "inet":
                out.setdefault(parts[1], []).append(parts[3])
        if out:
            return out
    try:  # fallback: psutil gives a netmask, so rebuild the prefix length
        import psutil

        for name, addrs in psutil.net_if_addrs().items():
            for a in addrs:
                if a.family != socket.AF_INET:
                    continue
                try:
                    prefix = ipaddress.IPv4Network(f"0.0.0.0/{a.netmask}").prefixlen
                except ValueError:
                    prefix = 24
                out.setdefault(name, []).append(f"{a.address}/{prefix}")
    except Exception:
        pass
    return out


def _nm_states() -> dict[str, str]:
    """``{device: nmcli state}``. Empty when NetworkManager is not installed."""
    if not shutil.which("nmcli"):
        return {}
    states = {}
    for line in _run(["nmcli", "-t", "-f", "DEVICE,STATE", "device"]).splitlines():
        # nmcli escapes colons inside MAC-named devices as "\:", so split from
        # the right on the single unescaped separator.
        dev, _, state = line.rpartition(":")
        if dev:
            states[dev.replace("\\:", ":")] = state
    return states


def wired_interfaces() -> list[dict]:
    """Every physical Ethernet port, with the state that decides the diagnosis."""

    def probe() -> list[dict]:
        try:
            names = sorted(os.listdir("/sys/class/net"))
        except OSError:
            return []
        addrs = _addrs_by_iface()
        nm = _nm_states()
        out = []
        for name in names:
            if not _is_physical_ethernet(name):
                continue
            flags = _sysfs(name, "flags")
            try:
                up = bool(int(flags, 16) & 0x1)  # IFF_UP
            except ValueError:
                up = None
            state = nm.get(name, "")
            out.append({
                "name": name,
                "up": up,
                # carrier reads back EINVAL while the link is down, hence "".
                "carrier": {"1": True, "0": False}.get(_sysfs(name, "carrier")),
                "operstate": _sysfs(name, "operstate"),
                "nm_state": state,
                "nm_managed": (None if not nm else state != "unmanaged"),
                "addresses": addrs.get(name, []),
            })
        return out

    return _cached("wired", probe)  # type: ignore[return-value]


def _wanted(cfg) -> list[tuple[str, str]]:
    """The (address, why-it-is-needed) pairs the wired NIC must carry."""
    sc = cfg.source
    return [
        (sc.host_addr, "this host on the sensor's subnet — SOPAS and the web UI"),
        (sc.udp_dest_addr, "the address the sensor streams Compact data to"),
        (sc.gateway_addr, "the sensor's gateway, so its UDP packets route here"),
    ]


def _pick_iface(cfg, ifaces: list[dict]) -> dict | None:
    """The port the sensor is most likely plugged into.

    Preference order: the configured name, then whichever port already carries
    one of the sensor addresses (it was set up before and only half survived),
    then a port with a cable in it, then the only candidate there is.
    """
    if not ifaces:
        return None
    by_name = {i["name"]: i for i in ifaces}
    configured = (cfg.source.wired_iface or "").strip()
    if configured:
        return by_name.get(configured)

    wanted_ips = {a.split("/")[0] for a, _ in _wanted(cfg)}
    for i in ifaces:
        if wanted_ips & {a.split("/")[0] for a in i["addresses"]}:
            return i
    linked = [i for i in ifaces if i["carrier"]]
    if len(linked) == 1:
        return linked[0]
    return linked[0] if linked else (ifaces[0] if len(ifaces) == 1 else None)


def commands_for(iface: str, cfg) -> list[dict]:
    """The repair sequence for ``iface``, each step with its own reason.

    Emitted whole rather than as "only the missing bits": the flush is what
    makes the sequence safe to re-run, and adding an address on top of a stale
    configuration is how you get ``RTNETLINK answers: File exists`` half-states.
    """
    steps = [
        (f"sudo nmcli device set {iface} managed no",
         "stop NetworkManager reconfiguring this port (no DHCP server on this cable)"),
        (f"sudo ip addr flush dev {iface}",
         "drop whatever half-configuration is on the port now"),
    ]
    for addr, why in _wanted(cfg):
        steps.append((f"sudo ip addr add {addr} dev {iface}", why))
    steps.append((f"sudo ip link set dev {iface} up", "bring the port back up"))
    return [{"cmd": c, "why": w} for c, w in steps]


def diagnose(cfg, streaming: bool | None = None) -> dict:
    """What is wrong with the wired link, and the script that fixes it.

    ``streaming`` is the sensor probe's verdict; when data is already arriving
    the interface is by definition fine, so nothing is suggested.
    """
    if not sys.platform.startswith("linux"):
        return {"supported": False, "suggested": False,
                "detail": "wired-link repair is only wired up for Linux"}

    ifaces = wired_interfaces()
    iface = _pick_iface(cfg, ifaces)
    wanted = _wanted(cfg)
    result: dict = {
        "supported": True,
        "interfaces": ifaces,
        "iface": iface["name"] if iface else None,
        "problems": [],
        "commands": [],
        "script": "",
        "suggested": False,
    }

    if iface is None:
        result["detail"] = (
            "no single wired port to blame — "
            + (f"pick one with source.wired_iface in config.yaml "
               f"({', '.join(i['name'] for i in ifaces)})" if ifaces
               else "this machine reports no physical Ethernet port")
        )
        return result

    have = {a.split("/")[0] for a in iface["addresses"]}
    # Addressing faults — the ones this script actually repairs.
    config_problems: list[str] = []
    if iface["up"] is False:
        config_problems.append("the port is administratively down")
    if iface["nm_managed"]:
        config_problems.append(f"NetworkManager still manages {iface['name']}, so it "
                               "will keep tearing static addresses back down")
    missing = [(a, why) for a, why in wanted if a.split("/")[0] not in have]
    for addr, why in missing:
        config_problems.append(f"{addr} is not on the port — {why}")

    # A dead carrier is a cable, not a configuration. Worth saying, never worth
    # fixing with `ip addr` — so it is reported but does not trigger the script.
    link_down = iface["carrier"] is False
    problems = list(config_problems)
    if link_down:
        problems.insert(0, "no carrier — nothing is plugged into this port, or the "
                           "cable/sensor is unpowered")

    result["problems"] = problems
    result["link_down"] = link_down
    result["missing"] = [a for a, _ in missing]
    result["commands"] = commands_for(iface["name"], cfg)
    result["script"] = "\n".join(
        f"# {s['why']}\n{s['cmd']}" for s in result["commands"]
    )
    # Suggest the fix when the addressing is wrong *and* no data is arriving. A
    # streaming sensor over an oddly-addressed NIC is not worth interrupting.
    result["suggested"] = bool(config_problems) and streaming is not True
    result["detail"] = (
        f"{iface['name']} is addressed for the sensor" if not config_problems
        else f"{len(config_problems)} addressing problem(s) on {iface['name']}"
    ) + (" (but no cable is plugged in)" if link_down else "")
    return result
