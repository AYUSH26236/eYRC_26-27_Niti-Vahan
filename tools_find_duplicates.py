"""
Lists functions/classes that are defined more than once in the same file
(the later definition silently replaces the earlier one), plus imports that
are never used.

Usage:  python find_duplicates.py task1a/*.py task1b/*.py task1c/*.py
"""

import ast
import collections
import sys


def check(path):
    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=path)

    # Only definitions in the SAME scope count: two classes may both have __init__.
    duplicates = {}
    for scope in ast.walk(tree):
        if not isinstance(scope, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        defined = collections.defaultdict(list)
        for node in scope.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined[node.name].append(node.lineno)
        for name, lines in defined.items():
            if len(lines) > 1:
                duplicates[name] = lines

    imported = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported[(alias.asname or alias.name).split(".")[0]] = node.lineno
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported[alias.asname or alias.name] = node.lineno
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    used |= {n.value.id for n in ast.walk(tree)
             if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
    unused = {name: line for name, line in imported.items() if name not in used}
    return duplicates, unused


def main(paths):
    for path in paths:
        duplicates, unused = check(path)
        print(path)
        for name, lines in sorted(duplicates.items()):
            print("  DUPLICATE  %-24s defined on lines %s" % (name, lines))
        for name, line in sorted(unused.items(), key=lambda item: item[1]):
            print("  UNUSED     import %-17s line %d" % (name, line))
        if not duplicates and not unused:
            print("  clean")


if __name__ == "__main__":
    main(sys.argv[1:])
