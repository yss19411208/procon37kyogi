"""公式の簡易版回答用サーバーをローカルで動かすための補助。

公式配布物 `procon-server-*` のバイナリを使います。ライセンスの都合で同梱は
していないので、公式サイトからダウンロードしたバイナリを `server/` に置いてください。

  https://www.procon.gr.jp/news/2026/08/20/117

設定 JSON の構造は同梱の「試合設定JSONについて.md」に従います。
`players` は teams の要素数から決まるので、設定には書きません。
"""

from __future__ import annotations

import json
import platform
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from hexa.problem import Problem

BINARY_NAMES = {
    ("Linux", "x86_64"): "procon-server-linux-amd64",
    ("Darwin", "arm64"): "procon-server-darwin-arm64",
    ("Darwin", "x86_64"): "procon-server-darwin-amd64",
    ("Windows", "AMD64"): "procon-server-windows-amd64.exe",
}


def find_server_binary(server_dir: Path) -> Path | None:
    """このOSに合ったバイナリを探す。見つからなければ None。"""

    expected = BINARY_NAMES.get((platform.system(), platform.machine()))
    if expected:
        candidate = server_dir / expected
        if candidate.exists():
            return candidate
    if not server_dir.exists():
        return None
    for candidate in sorted(server_dir.iterdir()):
        if candidate.is_file() and candidate.name.startswith("procon-server"):
            return candidate
    return None


def build_config(problem: Problem, tokens: list[str], names: list[str] | None = None) -> dict:
    rows = [
        list(problem.cells[row * problem.width:(row + 1) * problem.width])
        for row in range(problem.height)
    ]
    teams = []
    for index, token in enumerate(tokens):
        entry: dict[str, object] = {"token": token}
        if names and index < len(names):
            entry["name"] = names[index]
        teams.append(entry)
    return {
        "problem": {
            "width": problem.width,
            "height": problem.height,
            "cells": rows,
            "spots": [
                {"brand": spot.brand, "pos": spot.position, "stocks": spot.stocks}
                for spot in problem.spots
            ],
            "agentStarts": list(problem.agent_positions),
            "fuelLimits": problem.fuel_limit,
            "daySteps": list(problem.day_steps),
            "daySeconds": list(problem.day_seconds),
            "busyThreshold": problem.busy_threshold,
            "jammedThreshold": problem.jammed_threshold,
        },
        "teams": teams,
    }


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _address_rank(address: str) -> tuple:
    """別PCから繋ぐのに向いている順。小さいほど良い。"""

    if address.startswith("127."):
        return (9, address)          # 自分からしか見えない
    if address.startswith("169.254."):
        return (2, address)          # DHCPなし（識別されていないネットワーク）
    if (address.startswith("192.168.") or address.startswith("10.")
            or any(address.startswith(f"172.{block}.")
                   for block in range(16, 32))):
        return (0, address)          # ふつうのLAN
    return (1, address)              # それ以外（グローバルなど）


def local_ips() -> list[str]:
    """このPCのIPv4アドレスを、別PCから繋ぎやすい順に並べて返す。

    **なぜ1つでは足りないか**

    以前は「8.8.8.8 へ繋ぐふりをして自分のアドレスを調べる」方式
    だけでした。既定の経路が決まっていれば正しく出ますが、
    **LANケーブル直結でインターネットが無いPCでは失敗し、
    `127.0.0.1` を返していました**。その値を「別のPCから」として
    表示していたので、相手からは絶対に繋がりません。

    Windowsが「識別されていないネットワーク」と出すのは、DHCPが
    無くてアドレスが決まらない（169.254.x.x になる）状態です。
    まさにこの場合に当たります。

    そこで、経路から調べる方法に加えて**ホスト名からの逆引き**でも
    集め、重複を除いて並べます。候補を全部見せれば、どれかで繋がります。
    """

    found: list[str] = []

    def add(address: str | None) -> None:
        if not address or ":" in address:   # IPv6 はここでは扱わない
            return
        if address not in found:
            found.append(address)

    # 1) 既定の経路から（インターネットがあるときはこれが当たり）
    for probe in ("8.8.8.8", "192.168.1.1", "10.0.0.1"):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.settimeout(0.2)
                # 実際には送信しない。経路を決めるためだけに使う。
                sock.connect((probe, 80))
                add(sock.getsockname()[0])
        except OSError:
            pass
    # 2) ホスト名から（経路が無くても、挿してあるLANのアドレスが出る）
    try:
        add(socket.gethostbyname(socket.gethostname()))
    except OSError:
        pass
    try:
        for entry in socket.getaddrinfo(socket.gethostname(), None,
                                        socket.AF_INET):
            add(entry[4][0])
    except OSError:
        pass
    try:
        for address in socket.gethostbyname_ex(socket.gethostname())[2]:
            add(address)
    except OSError:
        pass
    found.sort(key=_address_rank)
    return found or ["127.0.0.1"]


def local_ip() -> str:
    """同じLANの別PCから見えるこのPCのアドレス（いちばん有望なもの）。

    見つからないときだけ `127.0.0.1` を返します。候補が複数あるときは
    `local_ips()` で全部出せます。
    """

    return local_ips()[0]


@dataclass
class LocalServer:
    process: subprocess.Popen
    base_url: str
    config_path: Path
    log_path: Path

    def read_log(self, limit: int = 4000) -> str:
        try:
            return self.log_path.read_text(encoding="utf-8", errors="replace")[-limit:]
        except OSError:
            return ""

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        handle = getattr(self.process, "_hexa_log_handle", None)
        if handle is not None and not handle.closed:
            handle.close()


def start_server(
    binary: Path,
    problem: Problem,
    tokens: list[str],
    work_dir: Path,
    names: list[str] | None = None,
    kind_deadline: str = "3s",
    match_start_delay: str = "2s",
    port: int | None = None,
    host: str = "127.0.0.1",
) -> LocalServer:
    work_dir.mkdir(parents=True, exist_ok=True)
    config_path = work_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(build_config(problem, tokens, names), handle, ensure_ascii=False)

    port = port or free_port()

    # サーバーの出力はパイプではなくファイルに逃がす。
    # パイプにしたまま親が読まずにいると、出力がバッファを埋めた時点で
    # サーバー側の書き込みが止まり、試合が進まなくなる（Windows で顕著）。
    log_path = work_dir / "server.log"
    log_handle = open(log_path, "w", encoding="utf-8", errors="replace")
    process = subprocess.Popen(
        [
            str(binary),
            "-config", str(config_path),
            # host に 0.0.0.0 を渡すと、同じLANの別PCからも接続できる。
            "-addr", f"{host}:{port}",
            "-kind-deadline", kind_deadline,
            "-match-start-delay", match_start_delay,
        ],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    process._hexa_log_handle = log_handle

    visible_host = local_ip() if host in ("0.0.0.0", "") else host
    base_url = f"http://{visible_host}:{port}"
    server = LocalServer(
        process=process, base_url=base_url, config_path=config_path, log_path=log_path
    )
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            log_handle.close()
            raise RuntimeError(f"サーバーが起動できませんでした:\n{server.read_log()}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.3):
                return server
        except OSError:
            time.sleep(0.15)
    server.stop()
    raise TimeoutError(f"サーバーが応答しませんでした:\n{server.read_log()}")
