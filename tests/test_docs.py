"""The CLI reference of the documentation (docs/cli.md) follows the CLI."""

import re
from importlib.metadata import entry_points
from pathlib import Path

import click
import pytest

CLI_MD = Path(__file__).resolve().parents[1] / "docs" / "cli.md"

#: The heading of a command's section: ### `instagraal-refine` { #instagraal-refine }
_SECTION = re.compile(r"^### `([a-z0-9_-]+)`", re.MULTILINE)
#: The first cell of an options table row: | `-i, --info-frags` | ... or | `--balance/--no-balance` | ...
_OPTION_CELL = re.compile(r"^\| `(-[^`]+)` \|", re.MULTILINE)

#: Options every command has or documents once, in the conventions of docs/cli.md.
_CONVENTIONS = {"-h", "--help", "-V", "--version"}


def commands():
    """The click commands of instaGRAAL's console scripts, by name."""
    scripts = entry_points(group="console_scripts")
    return {ep.name: ep.load() for ep in scripts if ep.value.startswith("instagraal.")}


def documented_options():
    """The options documented for each command, by command name."""
    text = CLI_MD.read_text(encoding="utf-8")
    headings = list(_SECTION.finditer(text))
    sections = {}
    for heading, following in zip(headings, [*headings[1:], None], strict=True):
        body = text[heading.end() : following.start() if following else len(text)]
        options = set()
        for cell in _OPTION_CELL.findall(body):
            options.update(name.strip() for name in re.split(r"[,/]", cell))
        sections[heading.group(1)] = options
    return sections


def cli_options(command):
    """The options of a click command, hidden ones included, without --help and --version."""
    options = set()
    for param in command.params:
        if isinstance(param, click.Option):
            options.update(param.opts + param.secondary_opts)
    return options - _CONVENTIONS


def test_every_command_is_documented():
    assert set(documented_options()) == set(commands())


@pytest.mark.parametrize("name", sorted(commands()))
def test_every_option_is_documented(name):
    documented = documented_options()[name]
    actual = cli_options(commands()[name])
    missing = actual - documented
    unknown = documented - actual
    assert not missing, f"options of `{name}` missing from docs/cli.md: {missing}"
    assert not unknown, f"docs/cli.md documents options `{name}` lacks: {unknown}"
