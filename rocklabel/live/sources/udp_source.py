"""Live point source for a SICK multiScan136 over UDP (Compact format).

A background thread receives UDP datagrams from the sensor, decodes each one
with :mod:`lidarrig.sources.compact_parser`, and pushes the resulting
:class:`PointBatch` onto a bounded queue.  When the queue is full the *oldest*
batch is dropped so the visualizer never falls behind the live stream.

Network setup (see the README) — on macOS give the Ethernet interface a manual
IPv4 such as ``192.168.0.100/24`` and point the sensor's UDP output at that
host:port via SOPAS / the sensor web interface.
"""

from __future__ import annotations

import queue
import socket
import threading
import time

import numpy as np

from rocklabel.live.config import AppConfig
from rocklabel.live.sources import compact_parser
from rocklabel.live.sources.base import PointBatch, PointSource

# multiScan136 Compact datagrams are large; size the recv buffer generously.
_RECV_BUFSIZE = 1 << 16  # 65536 bytes, the max UDP payload


class MultiScanUDPSource(PointSource):
    """Receive and decode live Compact-format scan data from a multiScan136."""

    def __init__(self, config: AppConfig) -> None:
        sc = config.source
        self._bind_host = sc.udp_bind_host
        self._port = int(sc.udp_port)
        self._sensor_ip = sc.sensor_ip
        self._queue: queue.Queue[PointBatch] = queue.Queue(maxsize=int(sc.udp_queue_size))
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Lightweight diagnostics readable by the app for the stats readout.
        self.packets_received = 0
        self.packets_dropped = 0
        self.parse_errors = 0
        self.imu_packets = 0

        # Latest IMU orientation quaternion (w, x, y, z), written by the rx
        # thread and attached to every scan batch. A single reference-swap is
        # atomic under the GIL, so no lock is needed.
        self._latest_quat: np.ndarray | None = None

    # -- PointSource contract ------------------------------------------------ #
    def start(self) -> None:
        if self._thread is not None:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)  # 8 MiB
        except OSError:
            pass  # best-effort; not fatal if the OS caps it lower
        sock.bind((self._bind_host, self._port))
        sock.settimeout(0.5)
        self._sock = sock

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._recv_loop, name="multiscan-udp-rx", daemon=True
        )
        self._thread.start()

    def read(self, timeout: float | None = None) -> PointBatch | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    # -- receiver thread ----------------------------------------------------- #
    def _recv_loop(self) -> None:
        assert self._sock is not None
        while not self._stop_event.is_set():
            try:
                data, _addr = self._sock.recvfrom(_RECV_BUFSIZE)
            except socket.timeout:
                continue
            except OSError:
                break  # socket closed during stop()

            if not data.startswith(compact_parser.START_OF_FRAME):
                # Not a Compact telegram (could be MSGPACK or noise); ignore.
                continue

            try:
                seg = compact_parser.parse_segment(data)
            except compact_parser.CompactParseError:
                self.parse_errors += 1
                continue

            self.packets_received += 1

            if seg.imu is not None:  # IMU telegram: update the pose, no points
                q = seg.imu.orientation
                if float(np.dot(q, q)) > 0.25:  # guard against all-zero quats
                    self._latest_quat = q
                    self.imu_packets += 1
                continue
            if seg.points.shape[0] == 0:
                continue

            batch = PointBatch(
                points=seg.points,
                intensity=seg.intensity,
                timestamp=time.time(),
                orientation=self._latest_quat,
            )
            self._push_drop_oldest(batch)

    def _push_drop_oldest(self, batch: PointBatch) -> None:
        """Enqueue ``batch``; if the queue is full, discard the oldest first so
        the consumer always sees the freshest data."""
        while True:
            try:
                self._queue.put_nowait(batch)
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                    self.packets_dropped += 1
                except queue.Empty:
                    # Raced with the consumer; loop and retry the put.
                    pass
