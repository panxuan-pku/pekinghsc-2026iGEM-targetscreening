"""Local launcher: validate service/workspace/PID, never stop an unrelated server."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import webbrowser
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


def probe(port, root):
    try:
        with urlopen(f"http://127.0.0.1:{port}/api/health", timeout=2) as response:
            body = json.load(response)
            valid = (response.status == 200 and isinstance(body, dict)
                     and body.get("service") == "VirtualCellTool"
                     and body.get("root") == str(root.resolve())
                     and body.get("ready") is True
                     and type(body.get("pid")) is int and body["pid"] > 0)
            if valid:
                return body
    except HTTPError as exc:
        raise RuntimeError(f"端口 {port} 返回 HTTP {exc.code}，不是已就绪的当前项目服务；请核查占用或旧版本服务。") from exc
    except URLError as exc:
        if isinstance(exc.reason, ConnectionRefusedError):
            return None
        raise RuntimeError(f"端口 {port} 无法确认服务身份：{exc.reason}") from exc
    except (ValueError, OSError, AttributeError) as exc:
        raise RuntimeError(f"端口 {port} 没有返回有效的服务身份：{exc}") from exc
    raise RuntimeError(f"端口 {port} 被其他服务/工作树占用，或当前服务尚未就绪；不会自动停止它。")


def launch(port, root, timeout=120):
    if not 1 <= port <= 65535:
        raise ValueError("VCT_PORT 必须在 1–65535 之间")
    existing = probe(port, root)
    if existing:
        return existing
    runtime = Path(os.environ.get("VCT_WORKSPACE", root.parent / "workspace" / "vct"))
    runtime.mkdir(parents=True, exist_ok=True)
    log_path = runtime / "vct_server.log"
    print(f"启动 VirtualCellTool（端口 {port}）；日志：{log_path}", flush=True)
    with log_path.open("ab") as log:
        child = subprocess.Popen([sys.executable, str(root / "web/app.py")], cwd=root,
                                 env=dict(os.environ, VCT_PORT=str(port)), stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if child.poll() is not None:
                raise RuntimeError(f"服务提前退出（{child.returncode}）；请查看 {log_path}")
            health = probe(port, root)
            if health:
                if health["pid"] != child.pid:
                    raise RuntimeError("启动期间端口被其他进程占用；拒绝把其他进程当成本次服务。")
                records = runtime / "runtime"
                records.mkdir(exist_ok=True)
                (records / f"server-{port}.json").write_text(
                    json.dumps(dict(health, port=port), ensure_ascii=False, indent=2) + "\n")
                return health
            time.sleep(1)
        raise RuntimeError(f"启动超时（{timeout}s）；请查看 {log_path}")
    except BaseException:
        # Only the child created by this invocation; never use PID-file contents to kill.
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        raise


def main():
    root = Path(__file__).resolve().parents[1]
    try:
        port = int(os.environ.get("VCT_PORT", "8377"))
        health = launch(port, root)
        print(f"✓ 当前项目服务已就绪：http://127.0.0.1:{port} · PID {health['pid']}")
        print("停止前请按 README 核对端口和进程；不要使用宽泛的 pkill 命令。")
        webbrowser.open(f"http://127.0.0.1:{port}")
        return 0
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"启动失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
