#!/usr/bin/env python3
"""Migrate benchmark verification payload capture to post-timing hook.

This script moves top-level `self._set_verification_payload(...)` statements out
of `benchmark_fn()` and into a new `capture_verification_payload()` method so
verification payload work happens post-timing.

Design notes:
- Only moves *top-level* statements in `benchmark_fn()` (not nested in if/with),
  to avoid creating empty blocks and breaking syntax.
- Skips classes that already define `capture_verification_payload`.
- Preserves the original `_set_verification_payload(...)` statement text (and
  indentation) to minimize behavioral changes.
"""

from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Migration:
    path: Path
    class_name: str
    benchmark_fn_line: int
    moved_stmt_range: Tuple[int, int]


def _iter_python_files(repo_root: Path, roots: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    for root in roots:
        base = repo_root / root
        if not base.exists():
            continue
        files.extend(p for p in base.rglob("*.py") if p.is_file())
    return sorted(set(files))


def _find_payload_stmt_in_benchmark_fn(fn: ast.FunctionDef) -> Optional[ast.stmt]:
    """Return the top-level statement calling _set_verification_payload(), if present."""
    for stmt in fn.body:
        if not isinstance(stmt, ast.Expr):
            continue
        call = stmt.value
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        if isinstance(func, ast.Attribute) and func.attr == "_set_verification_payload":
            return stmt
    return None


def _base_name(expr: ast.AST) -> Optional[str]:
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        return _base_name(expr.value)
    if isinstance(expr, ast.Call):
        return _base_name(expr.func)
    if isinstance(expr, ast.Subscript):
        return _base_name(expr.value)
    return None


def _payload_output_local_name(stmt: ast.stmt) -> Optional[str]:
    if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
        return None
    call = stmt.value
    for kw in call.keywords:
        if kw.arg == "output":
            name = _base_name(kw.value)
            if name and name != "self":
                return name
    return None


def _is_name_assigned(fn: ast.FunctionDef, name: str) -> bool:
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return True
        if isinstance(node, ast.AnnAssign):
            target = node.target
            if isinstance(target, ast.Name) and target.id == name:
                return True
        if isinstance(node, ast.AugAssign):
            target = node.target
            if isinstance(target, ast.Name) and target.id == name:
                return True
    return False


def _class_has_capture_method(cls: ast.ClassDef) -> bool:
    return any(isinstance(node, ast.FunctionDef) and node.name == "capture_verification_payload" for node in cls.body)


def _find_benchmark_fn(cls: ast.ClassDef) -> Optional[ast.FunctionDef]:
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == "benchmark_fn":
            return node
    return None


def _apply_migration_to_text(text: str, cls: ast.ClassDef, fn: ast.FunctionDef, stmt: ast.stmt) -> str:
    lines = text.splitlines(keepends=True)

    if stmt.lineno is None or stmt.end_lineno is None:
        raise RuntimeError("AST node missing lineno/end_lineno")
    if fn.end_lineno is None:
        raise RuntimeError("benchmark_fn missing end_lineno")

    # Extract the exact statement text (preserve indentation/formatting).
    start = stmt.lineno - 1
    end = stmt.end_lineno - 1
    stmt_lines = lines[start : end + 1]

    locals_used_in_stmt = sorted({
        node.id
        for node in ast.walk(stmt)
        if isinstance(node, ast.Name) and _is_name_assigned(fn, node.id)
    })

    # If the payload references a local output variable (e.g., output=out), ensure
    # benchmark_fn assigns `self.output = out` and rewrite the moved payload call
    # to use `self.output` so capture_verification_payload does not rely on locals.
    output_local = _payload_output_local_name(stmt)
    inserted_in_benchmark_fn = 0
    if output_local and _is_name_assigned(fn, output_local):
        leading_ws = re.match(r"^(\s*)", stmt_lines[0]).group(1)  # type: ignore[union-attr]
        lines.insert(start, f"{leading_ws}self.output = {output_local}\n")
        inserted_in_benchmark_fn = 1
        start += 1
        end += 1
        stmt_text = "".join(stmt_lines)
        pattern = re.compile(rf"(\boutput\s*=\s*){re.escape(output_local)}\b")
        stmt_text = pattern.sub(r"\1self.output", stmt_text)
        stmt_lines = stmt_text.splitlines(keepends=True)
        locals_used_in_stmt = [name for name in locals_used_in_stmt if name != output_local]

    # Persist any other benchmark_fn locals referenced by the payload call so the
    # moved statement can execute in capture_verification_payload().
    if locals_used_in_stmt:
        leading_ws = re.match(r"^(\s*)", stmt_lines[0]).group(1)  # type: ignore[union-attr]
        for name in locals_used_in_stmt:
            lines.insert(start, f"{leading_ws}self._payload_{name} = {name}\n")
            inserted_in_benchmark_fn += 1
            start += 1
            end += 1

    # Remove the statement from benchmark_fn.
    del lines[start : end + 1]

    # Insert capture_verification_payload method immediately after benchmark_fn.
    # Note: fn.end_lineno is based on original text, so adjust for deleted lines
    # if the deleted statement was *before* the function end.
    deleted_lines = (end - start + 1)
    insert_at = fn.end_lineno  # 1-based, insert after this line -> index == end_lineno
    if stmt.end_lineno <= fn.end_lineno:
        insert_at -= (deleted_lines - inserted_in_benchmark_fn)
    insert_idx = insert_at  # 0-based because list indices

    indent = " " * fn.col_offset
    method_header = f"{indent}def capture_verification_payload(self) -> None:\n"
    if not stmt_lines or not stmt_lines[0].startswith(indent + " " * 4):
        # Defensive: ensure the moved statement is indented as method body.
        raise RuntimeError("Refusing to migrate: moved statement indentation unexpected")

    new_block: List[str] = []
    # Ensure a blank line before the new method for readability.
    if insert_idx > 0 and lines[insert_idx - 1].strip():
        new_block.append("\n")
    new_block.append(method_header)
    for name in locals_used_in_stmt:
        new_block.append(f"{indent}{' ' * 4}{name} = self._payload_{name}\n")
    new_block.extend(stmt_lines)
    if not new_block[-1].endswith("\n"):
        new_block[-1] += "\n"

    lines[insert_idx:insert_idx] = new_block
    return "".join(lines)


def migrate_file(path: Path) -> Tuple[bool, List[Migration], Optional[str]]:
    """Return (changed, migrations, error)."""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        return False, [], f"read failed: {exc}"

    # Fast-path skip.
    if "_set_verification_payload" not in text:
        return False, [], None

    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return False, [], f"syntax error: {exc}"

    migrations: List[Migration] = []
    updated_text = text

    # Apply at most one migration per class, iterating from bottom to top so line offsets are stable.
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    classes_sorted = sorted(classes, key=lambda n: (n.lineno or 0), reverse=True)

    for cls in classes_sorted:
        if _class_has_capture_method(cls):
            continue

        fn = _find_benchmark_fn(cls)
        if fn is None:
            continue

        stmt = _find_payload_stmt_in_benchmark_fn(fn)
        if stmt is None:
            continue

        updated_text = _apply_migration_to_text(updated_text, cls, fn, stmt)
        migrations.append(
            Migration(
                path=path,
                class_name=cls.name,
                benchmark_fn_line=int(fn.lineno or 0),
                moved_stmt_range=(int(stmt.lineno or 0), int(stmt.end_lineno or 0)),
            )
        )

        # Re-parse to refresh line numbers for subsequent edits in this file.
        tree = ast.parse(updated_text)

    return (updated_text != text), migrations, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--roots", nargs="+", default=["ch01", "ch02", "ch03", "ch04", "ch05", "ch06", "ch07", "ch08", "ch09", "ch10", "ch11", "ch12", "ch13", "ch14", "ch15", "ch16", "ch17", "ch18", "ch19", "ch20", "labs"])
    parser.add_argument("--apply", action="store_true", help="Write changes to disk")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of files changed (0 = no limit)")
    args = parser.parse_args()

    repo_root: Path = args.repo_root.resolve()
    files = _iter_python_files(repo_root, args.roots)

    changed_files = 0
    all_migrations: List[Migration] = []
    errors: List[str] = []

    for path in files:
        changed, migrations, err = migrate_file(path)
        if err:
            errors.append(f"{path.relative_to(repo_root)}: {err}")
            continue
        if not changed:
            continue

        changed_files += 1
        all_migrations.extend(migrations)

        if args.apply:
            original_text = path.read_text(encoding="utf-8")
            tree = ast.parse(original_text)
            updated_text = original_text
            classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
            classes_sorted = sorted(classes, key=lambda n: (n.lineno or 0), reverse=True)
            for cls in classes_sorted:
                if _class_has_capture_method(cls):
                    continue
                fn = _find_benchmark_fn(cls)
                if fn is None:
                    continue
                stmt = _find_payload_stmt_in_benchmark_fn(fn)
                if stmt is None:
                    continue
                updated_text = _apply_migration_to_text(updated_text, cls, fn, stmt)
                tree = ast.parse(updated_text)
            path.write_text(updated_text, encoding="utf-8")

        if args.limit and changed_files >= args.limit:
            break

    print(f"Files changed: {changed_files}")
    if all_migrations:
        print(f"Migrations: {len(all_migrations)}")
        for mig in all_migrations[:25]:
            rel = mig.path.resolve()
            try:
                rel = mig.path.relative_to(repo_root)  # type: ignore[assignment]
            except Exception:
                pass
            print(f"  - {rel}: {mig.class_name} (benchmark_fn@{mig.benchmark_fn_line}) moved {mig.moved_stmt_range}")
        if len(all_migrations) > 25:
            print("  ...")

    if errors:
        print(f"Errors: {len(errors)}")
        for err in errors[:50]:
            print(f"  - {err}")
        if len(errors) > 50:
            print("  ...")

    if not args.apply:
        print("Dry run only. Re-run with --apply to write changes.")

    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
