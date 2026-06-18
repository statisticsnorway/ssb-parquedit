"""Nox sessions."""

import os
import shlex
import shutil
import sys
import tempfile
import tomllib
from pathlib import Path
from textwrap import dedent

import nox

try:
    from nox_poetry import Session
    from nox_poetry import session
except ImportError:
    message = f"""\
    Nox failed to import the 'nox-poetry' package.

    Please install it using the following command:

    {sys.executable} -m pip install nox-poetry"""
    raise SystemExit(dedent(message)) from None

package = "ssb_parquedit"
python_versions = ["3.13", "3.12", "3.14"]
python_versions_for_test = python_versions
nox.needs_version = ">= 2025.2.9"
nox.options.sessions = (
    "pre-commit",
    "mypy",
    "tests",
    "typeguard",
    "xdoctest",
    "docs-build",
)


def install_poetry_groups(session: Session, *groups: str) -> None:
    """Install dependencies from poetry groups, pinned to poetry.lock

    Using this as a workaround until this PR is merged in:
    https://github.com/cjolowicz/nox-poetry/pull/1080
    """

    def _load_dependency_groups() -> dict[str, list[object]]:
        pyproject_data = tomllib.loads(
            Path("pyproject.toml").read_text(encoding="utf-8")
        )
        groups_obj = pyproject_data.get("dependency-groups", {})
        return groups_obj if isinstance(groups_obj, dict) else {}

    def _resolve_group(
        group_name: str,
        all_groups: dict[str, list[object]],
        seen: set[str],
    ) -> list[str]:
        if group_name in seen:
            return []
        seen.add(group_name)

        resolved: list[str] = []
        for item in all_groups.get(group_name, []):
            if isinstance(item, str):
                resolved.append(item)
            elif isinstance(item, dict):
                include_group = item.get("include-group")
                if isinstance(include_group, str):
                    resolved.extend(_resolve_group(include_group, all_groups, seen))
        return resolved

    with tempfile.TemporaryDirectory() as tempdir:
        requirements_path = os.path.join(tempdir, "requirements.txt")

        # Prefer lockfile-pinned installs via poetry export when available.
        exported = True
        try:
            session.run(
                "poetry",
                "export",
                *[f"--only={group}" for group in groups],
                "--format=requirements.txt",
                "--without-hashes",
                f"--output={requirements_path}",
                external=True,
            )
        except nox.command.CommandFailed:
            exported = False

        if exported:
            # Use pip directly to avoid nox-poetry Session.install invoking poetry export again.
            session.run("python", "-m", "pip", "install", "-r", requirements_path)
            return

        all_groups = _load_dependency_groups()
        deps: list[str] = []
        for group in groups:
            deps.extend(_resolve_group(group, all_groups, set()))

        if deps:
            session.run("python", "-m", "pip", "install", *deps)


def activate_virtualenv_in_precommit_hooks(session: Session) -> None:
    """Activate virtualenv in hooks installed by pre-commit.

    This function patches git hooks installed by pre-commit to activate the
    session's virtual environment. This allows pre-commit to locate hooks in
    that environment when invoked from git.

    Args:
        session: The Session object.
    """
    assert session.bin is not None  # nosec

    # Only patch hooks containing a reference to this session's bindir. Support
    # quoting rules for Python and bash, but strip the outermost quotes so we
    # can detect paths within the bindir, like <bindir>/python.
    bindirs = [
        bindir[1:-1] if bindir[0] in "'\"" else bindir
        for bindir in (repr(session.bin), shlex.quote(session.bin))
    ]

    virtualenv = session.env.get("VIRTUAL_ENV")
    if virtualenv is None:
        return

    headers = {
        # pre-commit < 2.16.0
        "python": f"""\
            import os
            os.environ["VIRTUAL_ENV"] = {virtualenv!r}
            os.environ["PATH"] = os.pathsep.join((
                {session.bin!r},
                os.environ.get("PATH", ""),
            ))
            """,
        # pre-commit >= 2.16.0
        "bash": f"""\
            VIRTUAL_ENV={shlex.quote(virtualenv)}
            PATH={shlex.quote(session.bin)}"{os.pathsep}$PATH"
            """,
        # pre-commit >= 2.17.0 on Windows forces sh shebang
        "/bin/sh": f"""\
            VIRTUAL_ENV={shlex.quote(virtualenv)}
            PATH={shlex.quote(session.bin)}"{os.pathsep}$PATH"
            """,
    }

    hookdir = Path(".git") / "hooks"
    if not hookdir.is_dir():
        return

    for hook in hookdir.iterdir():
        if hook.name.endswith(".sample") or not hook.is_file():
            continue

        if not hook.read_bytes().startswith(b"#!"):
            continue

        text = hook.read_text()

        if not is_bindir_in_text(bindirs, text):
            continue

        lines = text.splitlines()
        hook.write_text(insert_header_in_hook(headers, lines))


def is_bindir_in_text(bindirs: list[str], text: str) -> bool:
    """Helper function to check if bindir is in text."""
    return any(
        Path("A") == Path("a") and bindir.lower() in text.lower() or bindir in text
        for bindir in bindirs
    )


def insert_header_in_hook(header: dict[str, str], lines: list[str]) -> str:
    """Helper function to insert headers in hook's text."""
    for executable, header_text in header.items():
        if executable in lines[0].lower():
            lines.insert(1, dedent(header_text))
            return "\n".join(lines)
    return "\n".join(lines)


@session(name="pre-commit", python=python_versions[0])
def precommit(session: Session) -> None:
    """Lint using pre-commit."""
    args = session.posargs or [
        "run",
        "--all-files",
        "--hook-stage=manual",
        "--show-diff-on-failure",
    ]
    install_poetry_groups(session, "lint")
    session.run("pre-commit", *args)
    if args and args[0] == "install":
        activate_virtualenv_in_precommit_hooks(session)


@session(python=python_versions)
def mypy(session: Session) -> None:
    """Type-check using mypy."""
    args = session.posargs or ["src", "tests"]
    session.install(".")
    install_poetry_groups(session, "dev")
    session.run("mypy", *args)
    if not session.posargs:
        session.run("mypy", f"--python-executable={sys.executable}", "noxfile.py")


@session(python=python_versions_for_test)
def tests(session: Session) -> None:
    """Run the test suite."""
    session.install(".")
    install_poetry_groups(session, "dev")
    try:
        session.run(
            "coverage",
            "run",
            "--parallel",
            "-m",
            "pytest",
            "-o",
            "pythonpath=",
            *session.posargs,
            "--ignore=tests/integration",
        )
    finally:
        if session.interactive:
            session.notify("coverage", posargs=[])


@session(python=python_versions[0])
def coverage(session: Session) -> None:
    """Produce the coverage report."""
    args = session.posargs or ["report", "--skip-empty"]
    install_poetry_groups(session, "dev")
    if not session.posargs and any(Path().glob(".coverage.*")):
        session.run("coverage", "combine")

    session.run("coverage", *args)


@session(python=python_versions[0])
def typeguard(session: Session) -> None:
    """Runtime type checking using Typeguard."""
    session.install(".")
    install_poetry_groups(session, "dev")
    session.run("pytest", f"--typeguard-packages={package}", *session.posargs)


@session(python=python_versions)
def xdoctest(session: Session) -> None:
    """Run examples with xdoctest."""
    if session.posargs:
        args = [package, *session.posargs]
    else:
        args = [f"--modname={package}", "--command=all"]
        if "FORCE_COLOR" in os.environ:
            args.append("--colored=1")

    session.install(".")
    install_poetry_groups(session, "dev")
    session.run("python", "-m", "xdoctest", *args)


@session(name="docs-build", python=python_versions[0])
def docs_build(session: Session) -> None:
    """Build the documentation."""
    args = session.posargs or ["docs", "docs/_build"]
    if not session.posargs and "FORCE_COLOR" in os.environ:
        args.insert(0, "--color")

    session.install(".")
    install_poetry_groups(session, "doc")

    build_dir = Path("docs", "_build")
    if build_dir.exists():
        shutil.rmtree(build_dir)

    session.run("sphinx-build", *args)


@session(python=python_versions[0])
def docs(session: Session) -> None:
    """Build and serve the documentation with live reloading on file changes."""
    args = session.posargs or ["--open-browser", "docs", "docs/_build"]
    session.install(".")
    install_poetry_groups(session, "doc")

    build_dir = Path("docs", "_build")
    if build_dir.exists():
        shutil.rmtree(build_dir)

    session.run("sphinx-autobuild", *args)
