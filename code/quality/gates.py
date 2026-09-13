"""Quality engineering gates. Run: ``python code/quality/gates.py`` (non-zero exit on failure).

Static gates (AST, no imports of engine code):
  * float-ban        no float literals, ``float()`` calls, or ``float`` annotations in the engine
  * determinism      no clocks, randomness, or uuid generation in the engine (logging excepted)
  * network-isolation no HTTP/socket clients or LLM SDKs outside the LLM gateway package
  * secret-scan      no credentials in tracked text files; ``.env`` must be gitignored

Runtime gates (import the engine):
  * schema-contracts every contract model is strict, frozen, extra-forbidding, error-hiding,
                     and every tool name has a registered contract
"""

import ast
import inspect
import json
import re
import subprocess
import sys
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

CODE_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
REPO_ROOT: Final[Path] = CODE_ROOT.parent
ENGINE_ROOT: Final[Path] = CODE_ROOT / "buy_or_wait"
LLM_GATEWAY_DIR: Final[str] = "llm"
CLOCK_ALLOWED_DIR: Final[str] = "observability"

BANNED_FLOAT_MODULES: Final[frozenset[str]] = frozenset({"math", "statistics", "numpy", "pandas"})
NONDETERMINISTIC_MODULES: Final[frozenset[str]] = frozenset({"random", "secrets", "uuid"})
NONDETERMINISTIC_CALLS: Final[frozenset[str]] = frozenset(
    {"now", "utcnow", "today", "time", "time_ns", "monotonic", "perf_counter", "urandom"}
)
NETWORK_MODULES: Final[frozenset[str]] = frozenset(
    {
        "requests",
        "httpx",
        "aiohttp",
        "urllib3",
        "socket",
        "ssl",
        "ftplib",
        "smtplib",
        "websocket",
        "websockets",
        "http",
        "urllib",
    }
)
LLM_SDK_MODULES: Final[frozenset[str]] = frozenset({"anthropic", "openai", "google"})
SECRET_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"\bsk-[A-Za-z0-9]{32,}"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{35}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?m)^[ \t]*(ANTHROPIC_API_KEY|GEMINI_API_KEY|OPENAI_API_KEY)[ \t]*=[ \t]*\S{12,}"),
)
TEXT_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".py", ".md", ".toml", ".txt", ".json", ".yaml", ".yml", ".cfg", ".ini", ".csv", ".example"}
)
SKIPPED_DIRS: Final[frozenset[str]] = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        ".cache",
        "media",
        "logs",
    }
)


@dataclass(frozen=True, slots=True)
class Violation:
    gate: str
    path: str
    line: int
    detail: str


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _python_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*.py")):
        if not SKIPPED_DIRS.intersection(path.parts):
            yield path


def _relative(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    return {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}


def _is_isinstance_type_argument(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    parent = parents.get(node)
    if isinstance(parent, ast.Tuple | ast.BinOp):
        parent = parents.get(parent)
    return (
        isinstance(parent, ast.Call)
        and isinstance(parent.func, ast.Name)
        and parent.func.id == "isinstance"
    )


def _imported_roots(node: ast.Import | ast.ImportFrom) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name.split(".")[0] for alias in node.names]
    if node.level or node.module is None:
        return []
    return [node.module.split(".")[0]]


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------
def check_float_ban(source: str, path: str) -> list[Violation]:
    tree = ast.parse(source, filename=path)
    parents = _parents(tree)
    violations: list[Violation] = []
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.Constant) and type(node.value) is float:
            violations.append(Violation("float-ban", path, line, "float literal"))
        elif isinstance(node, ast.Name) and node.id == "float":
            if not _is_isinstance_type_argument(node, parents):
                violations.append(
                    Violation("float-ban", path, line, "float used outside isinstance")
                )
        elif isinstance(node, ast.Import | ast.ImportFrom):
            for root in _imported_roots(node):
                if root in BANNED_FLOAT_MODULES:
                    violations.append(Violation("float-ban", path, line, f"imports {root}"))
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "round"
        ):
            violations.append(Violation("float-ban", path, line, "round(); use quantize_money"))
    return violations


def check_determinism(source: str, path: str) -> list[Violation]:
    tree = ast.parse(source, filename=path)
    violations: list[Violation] = []
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.Import | ast.ImportFrom):
            for root in _imported_roots(node):
                if root in NONDETERMINISTIC_MODULES:
                    violations.append(Violation("determinism", path, line, f"imports {root}"))
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in NONDETERMINISTIC_CALLS
        ):
            violations.append(Violation("determinism", path, line, f".{node.func.attr}() call"))
    return violations


def check_network_isolation(source: str, path: str, *, is_llm_gateway: bool) -> list[Violation]:
    tree = ast.parse(source, filename=path)
    violations: list[Violation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Import | ast.ImportFrom):
            continue
        for root in _imported_roots(node):
            if root in NETWORK_MODULES or (root in LLM_SDK_MODULES and not is_llm_gateway):
                violations.append(
                    Violation("network-isolation", path, node.lineno, f"imports {root}")
                )
    return violations


def scan_engine(root: Path = ENGINE_ROOT) -> list[Violation]:
    violations: list[Violation] = []
    for file_path in _python_files(root):
        rel_parts = file_path.relative_to(root).parts
        source = file_path.read_text(encoding="utf-8")
        path = _relative(file_path)
        violations.extend(check_float_ban(source, path))
        if rel_parts[0] != CLOCK_ALLOWED_DIR:
            violations.extend(check_determinism(source, path))
        violations.extend(
            check_network_isolation(source, path, is_llm_gateway=rel_parts[0] == LLM_GATEWAY_DIR)
        )
    return violations


def scan_secrets_in_text(text: str, path: str) -> list[Violation]:
    violations: list[Violation] = []
    for pattern in SECRET_PATTERNS:
        for match in pattern.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            violations.append(Violation("secret-scan", path, line, "credential-like value"))
    return violations


def _text_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if SKIPPED_DIRS.intersection(path.relative_to(root).parts) or not path.is_file():
            continue
        if path.name == ".env":
            continue  # the local secrets file is checked for ignore status instead
        if path.suffix in TEXT_SUFFIXES or path.name in {".gitignore", ".env.example"}:
            yield path


def check_secrets(root: Path = REPO_ROOT) -> list[Violation]:
    violations: list[Violation] = []
    for file_path in _text_files(root):
        text = file_path.read_text(encoding="utf-8", errors="replace")
        violations.extend(scan_secrets_in_text(text, _relative(file_path)))
    gitignore = root / ".gitignore"
    ignored = gitignore.is_file() and ".env" in gitignore.read_text(encoding="utf-8").split()
    if not ignored:
        violations.append(Violation("secret-scan", ".gitignore", 0, ".env is not gitignored"))
    elif (root / ".git").exists():
        result = subprocess.run(
            ["git", "-C", str(root), "check-ignore", "-q", ".env"],
            check=False,
            capture_output=True,
        )
        if result.returncode != 0:
            violations.append(Violation("secret-scan", ".env", 0, "git does not ignore .env"))
    return violations


def check_schema_contracts() -> list[Violation]:
    if str(CODE_ROOT) not in sys.path:
        sys.path.insert(0, str(CODE_ROOT))
    from pydantic import BaseModel

    from buy_or_wait.schemas import (
        audit,
        base,
        decision,
        entities,
        evidence,
        state,
        tools,
    )
    from buy_or_wait.schemas.enums import ToolName

    violations: list[Violation] = []
    required = {
        "strict": True,
        "frozen": True,
        "extra": "forbid",
        "hide_input_in_errors": True,
    }
    modules = (base, entities, decision, evidence, audit, tools, state)
    for module in modules:
        for name, model in inspect.getmembers(module, inspect.isclass):
            if not issubclass(model, BaseModel) or model.__module__ != module.__name__:
                continue
            for key, expected in required.items():
                if model.model_config.get(key) != expected:
                    violations.append(
                        Violation(
                            "schema-contracts", module.__name__, 0, f"{name}.{key} != {expected}"
                        )
                    )
            for field_name, field in model.model_fields.items():
                if "float" in repr(field.annotation):
                    violations.append(
                        Violation(
                            "schema-contracts",
                            module.__name__,
                            0,
                            f"{name}.{field_name} is float-typed",
                        )
                    )
    missing = sorted(set(ToolName) - set(tools.TOOL_CONTRACTS))
    if missing:
        violations.append(
            Violation("schema-contracts", tools.__name__, 0, f"tools without contracts: {missing}")
        )
    return violations


def run_all() -> list[Violation]:
    return [*scan_engine(), *check_secrets(), *check_schema_contracts()]


def main() -> int:
    violations = run_all()
    report = {
        "passed": not violations,
        "violation_count": len(violations),
        "violations": [asdict(violation) for violation in violations],
    }
    sys.stdout.write(json.dumps(report, indent=2) + "\n")
    return 0 if not violations else 1


if __name__ == "__main__":
    raise SystemExit(main())
