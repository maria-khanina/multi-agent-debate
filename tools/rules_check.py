import ast
import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1])
ALLOWED_NUMBERS = {0, 1, -1, 1.0, 2.0}
ABBREVIATIONS = {
    "impl", "spec", "specs", "cfg", "idx", "num", "val", "obj", "fn", "ctx",
    "tmp", "arg", "args", "kwargs", "rng", "msg", "res", "resp", "req", "ep",
    "arch", "pv", "fmt", "attr", "attrs", "param", "params", "info", "doc",
    "docs", "lib", "repo", "dir", "dirs", "cls", "func", "var", "vars", "len",
}
STDLIB = set(sys.stdlib_module_names)
MAX_STATEMENTS = 10
CONFIG_FILE_SUFFIX = "_settings.py"


def find_project_packages(root):
    root = root.resolve()
    bases = [root, root.parent, root / "src", root.parent / "src"]
    packages = {marker.parent.name for base in bases for marker in base.glob("*/__init__.py")}
    if (root / "__init__.py").exists():
        packages.add(root.resolve().name)
    return packages


PROJECT_PACKAGES = find_project_packages(ROOT)


def count_statements(node: ast.AST) -> int:
    total = 0
    for child in ast.walk(node):
        if isinstance(child, ast.stmt) and child is not node:
            total += 1
    return total


def has_docstring(node: ast.FunctionDef) -> bool:
    return bool(node.body) and isinstance(node.body[0], ast.Expr) and isinstance(
        getattr(node.body[0], "value", None), ast.Constant
    ) and isinstance(node.body[0].value.value, str)


def check_function_sizes(path, tree, report):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            size = count_statements(node) - (1 if has_docstring(node) else 0)
            if size > MAX_STATEMENTS:
                report.append(f"{path}:{node.lineno} {node.name} has {size} statements")


def check_names(path, tree, report):
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Name):
            names.append(node.id)
        elif isinstance(node, ast.arg):
            names.append(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.append(node.name)
        for name in names:
            if len(name) == 1 and name != "_":
                report.append(f"{path}:{node.lineno} single-letter name {name}")
            tokens = set(name.lower().split("_"))
            bad = tokens & ABBREVIATIONS
            if bad and name not in {"len"}:
                report.append(f"{path}:{node.lineno} abbreviation {sorted(bad)} in {name}")
            if name.lower().endswith(("spec", "specs")) and name not in {"spec", "specs"}:
                report.append(f"{path}:{node.lineno} abbreviation Spec in {name}")


def check_defaults(path, tree, report):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.args.defaults or any(d is not None for d in node.args.kw_defaults):
                report.append(f"{path}:{node.lineno} default parameter in {node.name}")


def check_numbers(path, tree, report):
    if path.name.endswith(CONFIG_FILE_SUFFIX):
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            if isinstance(node.value, bool):
                continue
            if node.value not in ALLOWED_NUMBERS:
                report.append(f"{path}:{node.lineno} number {node.value}")


def module_group(name):
    root = name.split(".")[0]
    if root in STDLIB:
        return 0
    if root in PROJECT_PACKAGES:
        return 2
    return 1


def check_imports(path, tree, report):
    entries = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            entries.append((module_group(node.names[0].name), node.names[0].name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            entries.append((module_group(node.module or ""), node.module or "", node.lineno))
    for first, second in zip(entries, entries[1:]):
        if (first[0], first[1].lower()) > (second[0], second[1].lower()):
            report.append(f"{path}:{second[2]} import order {first[1]} before {second[1]}")
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            names = [alias.name for alias in node.names]
            if names != sorted(names, key=lambda n: (not n.isupper(), n[0].islower(), n)):
                report.append(f"{path}:{node.lineno} import names order {names}")


def check_container_hints(path, tree, report):
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            value = node.value
            is_container = isinstance(value, (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp))
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
                is_container = is_container or value.func.id in {"list", "dict", "set", "defaultdict"}
            if is_container and isinstance(node.targets[0], ast.Name):
                is_module_level = node in tree.body
                if not is_module_level:
                    report.append(f"{path}:{node.lineno} container without type hint: {node.targets[0].id}")


def main():
    report = []
    for path in sorted(ROOT.rglob("*.py")):
        if {"typings", "__pycache__", ".venv", "venv"} & set(path.parts):
            continue
        if path.resolve() == pathlib.Path(__file__).resolve():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for check in (check_function_sizes, check_names, check_defaults, check_numbers, check_imports, check_container_hints):
            check(path.relative_to(ROOT), tree, report)
    print("\n".join(report))
    print(len(report), "findings")


main()
