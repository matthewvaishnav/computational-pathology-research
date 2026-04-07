from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None


JSONRPC_VERSION = "2.0"
SERVER_NAME = "computational-pathology"
SERVER_VERSION = "0.1.0"
SUPPORTED_PROTOCOL_VERSIONS = (
    "2025-11-25",
    "2025-06-18",
    "2025-03-26",
    "2024-11-05",
)
SKIP_DIRECTORIES = {
    ".git",
    ".kilo",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "checkpoints",
    "checkpoints_demo",
    "data",
    "htmlcov",
    "logs",
    "results",
}
MAX_TEXT_FILE_BYTES = 1_000_000


class JSONRPCError(Exception):
    def __init__(self, code: int, message: str, data: Optional[Any] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


def _json_line(payload: Any) -> str:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


def _stringify(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, indent=2, sort_keys=True)


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def _is_probably_text_file(path: Path) -> bool:
    try:
        with open(path, "rb") as handle:
            sample = handle.read(4096)
    except OSError:
        return False
    return b"\x00" not in sample


class ProjectMCPServer:
    def __init__(self, project_root: Path):
        self.project_root = project_root.resolve()
        self.initialized = False
        self._tool_handlers: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
            "project_overview": self._tool_project_overview,
            "list_project_files": self._tool_list_project_files,
            "read_text_file": self._tool_read_text_file,
            "search_repository": self._tool_search_repository,
            "run_pytest": self._tool_run_pytest,
        }

    def run(self) -> int:
        for raw_line in sys.stdin.buffer:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError as exc:
                self._emit(self._error_response(None, -32700, f"Parse error: {exc}"))
                continue

            response = self.handle_message(message)
            if response is None:
                continue
            if isinstance(response, list):
                if response:
                    self._emit(response)
            else:
                self._emit(response)
        return 0

    def handle_message(self, message: Any) -> Optional[Any]:
        if isinstance(message, list):
            responses = []
            for item in message:
                response = self.handle_message(item)
                if response is None:
                    continue
                if isinstance(response, list):
                    responses.extend(response)
                else:
                    responses.append(response)
            return responses

        if not isinstance(message, dict):
            return self._error_response(None, -32600, "Invalid Request")
        if "method" not in message:
            return None

        method = message["method"]
        params = message.get("params", {})
        request_id = message.get("id")
        is_notification = "id" not in message

        try:
            result = self._dispatch(method, params)
        except JSONRPCError as exc:
            if is_notification:
                return None
            return self._error_response(request_id, exc.code, exc.message, exc.data)
        except Exception as exc:  # pragma: no cover
            if is_notification:
                return None
            return self._error_response(request_id, -32603, f"Internal error: {exc}")

        if is_notification:
            return None
        return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "result": result}

    def _dispatch(self, method: str, params: Any) -> Dict[str, Any]:
        if params is None:
            params = {}
        if method == "initialize":
            return self._handle_initialize(params)
        if method == "notifications/initialized":
            self.initialized = True
            return {}
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": self._tool_descriptions()}
        if method == "tools/call":
            return self._handle_tool_call(params)
        if method == "resources/list":
            return {"resources": []}
        if method == "prompts/list":
            return {"prompts": []}
        raise JSONRPCError(-32601, f"Method not found: {method}")

    def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        requested = params.get("protocolVersion")
        if requested in SUPPORTED_PROTOCOL_VERSIONS:
            protocol_version = requested
        else:
            protocol_version = SUPPORTED_PROTOCOL_VERSIONS[0]
        return {
            "protocolVersion": protocol_version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            "instructions": (
                "Use this server for repo-scoped inspection and targeted pytest "
                "runs inside the computational pathology workspace."
            ),
        }

    def _handle_tool_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments", {}) or {}
        if not isinstance(arguments, dict):
            raise JSONRPCError(-32602, "Tool arguments must be an object")
        if name not in self._tool_handlers:
            raise JSONRPCError(-32602, f"Unknown tool: {name}")
        return self._tool_handlers[name](arguments)

    def _tool_descriptions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "project_overview",
                "description": "Summarize the repo, key directories, and core metadata.",
                "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
            },
            {
                "name": "list_project_files",
                "description": "List files in the repository under a relative directory and glob.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "subdirectory": {"type": "string", "default": "."},
                        "glob": {"type": "string", "default": "*"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 100},
                    },
                    "additionalProperties": False,
                },
            },
            {
                "name": "read_text_file",
                "description": "Read a UTF-8 text file from the repository with optional line slicing.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer", "minimum": 1, "default": 1},
                        "end_line": {"type": "integer", "minimum": 1},
                        "max_chars": {"type": "integer", "minimum": 200, "maximum": 50000, "default": 12000},
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "search_repository",
                "description": "Search repository text files for a substring match.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "subdirectory": {"type": "string", "default": "."},
                        "case_sensitive": {"type": "boolean", "default": False},
                        "max_results": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "run_pytest",
                "description": "Run a targeted pytest command inside the repository.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "targets": {"type": "array", "items": {"type": "string"}},
                        "max_failures": {"type": "integer", "minimum": 1, "maximum": 20, "default": 1},
                        "timeout_seconds": {"type": "integer", "minimum": 5, "maximum": 1800, "default": 120},
                    },
                    "additionalProperties": False,
                },
            },
        ]

    def _tool_project_overview(self, _: Dict[str, Any]) -> Dict[str, Any]:
        overview = {
            "project_root": str(self.project_root),
            "project_name": self.project_root.name,
            "python_files": self._count_files("*.py"),
            "test_files": self._count_files("test_*.py", subdirectory="tests"),
            "config_files": (
                self._count_files("*.yaml")
                + self._count_files("*.yml")
                + self._count_files("*.json")
            ),
            "top_level_directories": self._top_level_directories(),
            "package_metadata": self._load_pyproject_metadata(),
        }
        return self._tool_result(overview)

    def _tool_list_project_files(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        subdirectory = arguments.get("subdirectory", ".")
        pattern = arguments.get("glob", "*")
        limit = int(arguments.get("limit", 100))
        directory = self._resolve_repo_path(subdirectory, must_exist=True, allow_directory=True)
        if not directory.is_dir():
            raise JSONRPCError(-32602, f"Not a directory: {subdirectory}")

        results: List[str] = []
        for path in sorted(directory.glob(pattern)):
            if path.is_dir() or self._path_is_skipped(path):
                continue
            results.append(self._relative_path(path))
            if len(results) >= limit:
                break

        payload = {
            "subdirectory": self._relative_path(directory),
            "glob": pattern,
            "count": len(results),
            "files": results,
        }
        return self._tool_result(payload)

    def _tool_read_text_file(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        relative_path = arguments["path"]
        start_line = int(arguments.get("start_line", 1))
        end_line = arguments.get("end_line")
        max_chars = int(arguments.get("max_chars", 12000))
        path = self._resolve_repo_path(relative_path, must_exist=True, allow_directory=False)
        if path.is_dir():
            raise JSONRPCError(-32602, f"Expected a file, got directory: {relative_path}")
        if path.stat().st_size > MAX_TEXT_FILE_BYTES:
            raise JSONRPCError(-32602, f"File is too large to read safely: {relative_path}")
        if not _is_probably_text_file(path):
            raise JSONRPCError(-32602, f"File does not appear to be a text file: {relative_path}")

        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        if start_line < 1:
            raise JSONRPCError(-32602, "start_line must be >= 1")
        if end_line is None:
            end_line = min(len(lines), start_line + 199)
        else:
            end_line = int(end_line)
        if end_line < start_line:
            raise JSONRPCError(-32602, "end_line must be >= start_line")

        selected = lines[start_line - 1:end_line]
        numbered = "\n".join(
            f"{line_number}: {line_text}"
            for line_number, line_text in enumerate(selected, start=start_line)
        )
        body = (
            f"Path: {self._relative_path(path)}\n"
            f"Lines: {start_line}-{start_line + len(selected) - 1} of {len(lines)}\n\n"
            f"{numbered}"
        )
        return self._tool_result(self._truncate_text(body, max_chars))

    def _tool_search_repository(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        query = arguments["query"]
        subdirectory = arguments.get("subdirectory", ".")
        case_sensitive = bool(arguments.get("case_sensitive", False))
        max_results = int(arguments.get("max_results", 20))
        directory = self._resolve_repo_path(subdirectory, must_exist=True, allow_directory=True)
        if not directory.is_dir():
            raise JSONRPCError(-32602, f"Not a directory: {subdirectory}")
        if not query:
            raise JSONRPCError(-32602, "query must be a non-empty string")

        haystack_query = query if case_sensitive else query.lower()
        matches: List[Dict[str, Any]] = []

        for file_path in self._iter_repo_files(directory):
            if file_path.stat().st_size > MAX_TEXT_FILE_BYTES or not _is_probably_text_file(file_path):
                continue
            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            for index, line in enumerate(text.splitlines(), start=1):
                candidate = line if case_sensitive else line.lower()
                if haystack_query not in candidate:
                    continue
                matches.append(
                    {"path": self._relative_path(file_path), "line": index, "text": line.strip()}
                )
                if len(matches) >= max_results:
                    break
            if len(matches) >= max_results:
                break

        payload = {
            "query": query,
            "subdirectory": self._relative_path(directory),
            "count": len(matches),
            "matches": matches,
        }
        return self._tool_result(payload)

    def _tool_run_pytest(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        targets = arguments.get("targets") or ["tests"]
        if not isinstance(targets, list) or not all(isinstance(item, str) for item in targets):
            raise JSONRPCError(-32602, "targets must be an array of relative paths")

        max_failures = int(arguments.get("max_failures", 1))
        timeout_seconds = int(arguments.get("timeout_seconds", 120))
        validated_targets = [
            str(self._resolve_repo_path(target, must_exist=True, allow_directory=True))
            for target in targets
        ]
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            f"--maxfail={max_failures}",
            *validated_targets,
        ]

        try:
            completed = subprocess.run(
                command,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return self._tool_error(
                f"pytest timed out after {timeout_seconds} seconds",
                {
                    "targets": [self._relative_path(Path(path)) for path in validated_targets],
                    "partial_stdout": self._truncate_text(exc.stdout or "", 12000),
                    "partial_stderr": self._truncate_text(exc.stderr or "", 12000),
                },
            )

        payload = {
            "command": command,
            "targets": [self._relative_path(Path(path)) for path in validated_targets],
            "exit_code": completed.returncode,
            "stdout": self._truncate_text(completed.stdout, 12000),
            "stderr": self._truncate_text(completed.stderr, 12000),
        }
        return self._tool_result(payload)

    def _tool_result(self, payload: Any) -> Dict[str, Any]:
        return {"content": [{"type": "text", "text": _stringify(payload)}]}

    def _tool_error(self, message: str, payload: Optional[Any] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {"message": message}
        if payload is not None:
            body["details"] = payload
        return {"isError": True, "content": [{"type": "text", "text": _stringify(body)}]}

    def _resolve_repo_path(
        self,
        path_value: str,
        *,
        must_exist: bool,
        allow_directory: bool,
    ) -> Path:
        candidate = Path(path_value)
        if not candidate.is_absolute():
            candidate = self.project_root / candidate
        candidate = candidate.resolve()

        if not _is_relative_to(candidate, self.project_root):
            raise JSONRPCError(-32602, f"Path escapes the repository root: {path_value}")
        if must_exist and not candidate.exists():
            raise JSONRPCError(-32602, f"Path does not exist: {path_value}")
        if not allow_directory and candidate.exists() and candidate.is_dir():
            raise JSONRPCError(-32602, f"Expected a file path: {path_value}")
        return candidate

    def _relative_path(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.project_root).as_posix()
        except ValueError:
            return str(path.resolve())

    def _path_is_skipped(self, path: Path) -> bool:
        return any(part in SKIP_DIRECTORIES for part in path.parts)

    def _iter_repo_files(self, start_directory: Path) -> Iterable[Path]:
        for current_root, dirnames, filenames in os.walk(start_directory):
            dirnames[:] = [name for name in dirnames if name not in SKIP_DIRECTORIES]
            for filename in filenames:
                path = Path(current_root) / filename
                if not self._path_is_skipped(path):
                    yield path

    def _count_files(self, pattern: str, subdirectory: str = ".") -> int:
        directory = self._resolve_repo_path(subdirectory, must_exist=True, allow_directory=True)
        return sum(
            1
            for path in directory.rglob(pattern)
            if path.is_file() and not self._path_is_skipped(path)
        )

    def _top_level_directories(self) -> List[str]:
        return [
            path.name
            for path in sorted(self.project_root.iterdir())
            if path.is_dir() and path.name not in {".git", ".venv", "__pycache__"}
        ]

    def _load_pyproject_metadata(self) -> Dict[str, Any]:
        pyproject_path = self.project_root / "pyproject.toml"
        if not pyproject_path.exists() or tomllib is None:
            return {}

        with open(pyproject_path, "rb") as handle:
            data = tomllib.load(handle)

        project = data.get("project", {})
        return {
            "name": project.get("name"),
            "version": project.get("version"),
            "requires_python": project.get("requires-python"),
            "description": project.get("description"),
        }

    def _truncate_text(self, text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        extra = len(text) - limit
        return f"{text[:limit]}\n\n... truncated {extra} characters ..."

    def _emit(self, payload: Any) -> None:
        sys.stdout.write(_json_line(payload))
        sys.stdout.write("\n")
        sys.stdout.flush()

    def _error_response(
        self,
        request_id: Optional[Any],
        code: int,
        message: str,
        data: Optional[Any] = None,
    ) -> Dict[str, Any]:
        error: Dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            error["data"] = data
        return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "error": error}


def main(argv: Optional[List[str]] = None) -> int:
    default_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Computational pathology MCP server")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=default_root,
        help="Repository root to expose through MCP tools.",
    )
    args = parser.parse_args(argv)
    server = ProjectMCPServer(args.project_root)
    return server.run()


if __name__ == "__main__":
    raise SystemExit(main())
