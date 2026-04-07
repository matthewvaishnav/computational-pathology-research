import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.mcp_server import JSONRPCError, ProjectMCPServer

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVER_SCRIPT = REPO_ROOT / "scripts" / "project_mcp_server.py"


def _run_server(messages):
    payload = "\n".join(json.dumps(message) for message in messages) + "\n"
    completed = subprocess.run(
        [sys.executable, str(SERVER_SCRIPT)],
        input=payload,
        text=True,
        capture_output=True,
        cwd=str(REPO_ROOT),
        timeout=20,
        check=False,
    )
    responses = [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
    return completed, responses


def test_stdio_round_trip_lists_tools():
    completed, responses = _run_server(
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2025-11-25"},
            },
            {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "project_overview", "arguments": {}},
            },
        ]
    )

    assert completed.returncode == 0
    assert [response["id"] for response in responses] == [1, 2, 3]
    assert responses[0]["result"]["serverInfo"]["name"] == "computational-pathology"
    tool_names = {tool["name"] for tool in responses[1]["result"]["tools"]}
    assert "project_overview" in tool_names
    assert "run_pytest" in tool_names
    assert "computational-pathology-research" in responses[2]["result"]["content"][0]["text"]


def test_read_text_file_stays_inside_repo():
    server = ProjectMCPServer(REPO_ROOT)
    response = server._tool_read_text_file({"path": "README.md"})
    assert "README.md" in response["content"][0]["text"]

    escaping_path = Path("..") / "outside.txt"
    # Use regex pattern that matches both Windows and Unix path separators
    with pytest.raises(JSONRPCError, match=r"escapes the repository root"):
        server._tool_read_text_file({"path": str(escaping_path)})


def test_run_pytest_builds_targeted_command(monkeypatch):
    server = ProjectMCPServer(REPO_ROOT)
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["cwd"] = kwargs["cwd"]

        class Result:
            returncode = 0
            stdout = "1 passed"
            stderr = ""

        return Result()

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = server._tool_run_pytest({"targets": ["tests/test_mcp_server.py"]})

    assert captured["command"][:3] == [sys.executable, "-m", "pytest"]
    assert captured["cwd"] == str(REPO_ROOT)
    assert "tests/test_mcp_server.py" in result["content"][0]["text"]
