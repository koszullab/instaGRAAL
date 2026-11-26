# Contributing

Bug reports, suggestions, documentation and pull requests are welcome on
[GitHub](https://github.com/koszullab/instaGRAAL). We follow the
[fork and pull model](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/getting-started/about-collaborative-development-models),
[conventional commits](https://www.conventionalcommits.org/en/v1.0.0/) and
[semantic versioning](https://semver.org).

## Development set-up

The environment is managed by [uv](https://docs.astral.sh/uv/). Install uv,
then, from a clone of your fork:

```bash
git clone https://github.com/koszullab/instaGRAAL.git
cd instaGRAAL
# instaGRAAL in editable mode, with the test and dev tools
uv sync --python 3.12 --extra dev
# or, with the CUDA toolkit, pycuda too
uv sync --python 3.12 --extra dev --extra gpu
# optional: the pre-commit hooks, on every commit
uv run pre-commit install
```

`uv run instagraal-endtoend --version` runs the version in the repository.
Without the `gpu` extra, everything but the MCMC scaffolder works.

## Tests, lint and format

```bash
uv run pytest                 # the tests; the GPU tests skip without a GPU
uv run pytest --no-gpu        # skip the GPU tests explicitly
uv run ruff check .           # lint
uv run ruff format .          # format (line length 130)
uv run mypy src               # type check (not run by CI)
```

- New functions come with tests in `tests/`. The test data, in `tests/data/`,
  is an assembly of 146 contigs cut from the yeast genome and 1.3 million
  Hi-C pairs on it: `scripts/generate_test_data.sh` makes them, and they are
  archived on Zenodo ([10.5281/zenodo.19813387](https://doi.org/10.5281/zenodo.19813387)).
  The `Makefile` runs the steps one by one on them (`make all`, with a GPU).
- [test-cpu.yml](https://github.com/koszullab/instaGRAAL/blob/master/.github/workflows/test-cpu.yml)
  runs the tests on Python 3.10 to 3.12, without pycuda, on every push and
  pull request.
- [test-gpu.yml](https://github.com/koszullab/instaGRAAL/blob/master/.github/workflows/test-gpu.yml)
  runs `tests/test_instagraal_gpu.py` on GPU pods rented from RunPod, for each
  Python version and CUDA 12.1 and 12.4: on pull requests that change the
  sources, the GPU tests, `conftest.py` or the test data; on pushes whose
  commit message holds `[gpu]`; and by hand. Pods are billed until they stop:
  the workflow and `gpu-pod-reaper.yml` terminate them.
- Docstrings follow the [numpydoc standard](https://numpydoc.readthedocs.io/en/latest/format.html):
  they make the [Python API reference](api/pre.md).

## Documentation

The site is built from `docs/` by [Zensical](https://zensical.org) and served
at <https://koszullab.github.io/instagraal/>.

```bash
uv run --only-group docs zensical serve    # preview: localhost:8000
uv run --only-group docs zensical build    # the site, in site/
```

The `docs` dependency group installs Zensical and mkdocstrings alone, without
instaGRAAL or pycuda.

- **Pages**: the Markdown files of `docs/`, listed in order by `nav` in
  `zensical.toml`. A new page goes in both.
- **Python API reference**: the pages of `docs/api/` hold
  `::: instagraal.<module>` directives, which
  [mkdocstrings](https://mkdocstrings.github.io) expands from the docstrings,
  read from `src/` without importing them.
- **CLI reference**: `docs/cli.md`, written by hand. `tests/test_docs.py`
  fails when it misses a command or option of the CLI, or documents one the
  CLI lacks.
- **Diagrams**: [Mermaid](https://mermaid.js.org) code blocks. **Images**:
  `docs/images/`.
- **Look**: [chromokit](https://github.com/js2264/chromokit), the UI kit of
  metaTOR's and metacooler's documentation. Do not edit its files
  (`overrides/`, `docs/stylesheets/chromokit.css`,
  `docs/javascripts/chromokit.js`, listed in `chromokit.json`); put
  instaGRAAL's own styles in `docs/stylesheets/site.css`. To update the kit,
  and check that no kit file was edited:

    ```bash
    npx github:js2264/chromokit sync
    npx github:js2264/chromokit check
    ```

Every push to `master` builds and deploys the site
([docs.yml](https://github.com/koszullab/instaGRAAL/blob/master/.github/workflows/docs.yml));
pull requests build it without deploying. The build is strict: any warning,
such as a broken link, fails it.

## Changelog

Every user-visible change gets an entry under `## Unreleased` in
`docs/changelog.md`, in *Breaking changes*, *Added*, *Changed*, *Deprecated*,
*Removed* or *Fixed*.

## Releasing

Releases use [bump-my-version](https://github.com/callowayproject/bump-my-version),
configured in `pyproject.toml`. From an up-to-date `master` with green CI:

```bash
uv run bump-my-version bump patch   # or minor / major
git push --follow-tags
```

The bump sets `version` in `pyproject.toml` (the single source of the
version), `release` in `zensical.toml` (the version the site shows) and the
`Version` label of the `Dockerfile`, turns `## Unreleased` in
`docs/changelog.md` into a dated section, commits as `chore(release): vX.Y.Z`
and tags `vX.Y.Z`. Preview it with
`uv run bump-my-version bump patch --dry-run -vv`.

Pushing the tag starts:

- [cd.yml](https://github.com/koszullab/instaGRAAL/blob/master/.github/workflows/cd.yml),
  which builds the sdist and wheel and publishes them to PyPI;
- [docker.yml](https://github.com/koszullab/instaGRAAL/blob/master/.github/workflows/docker.yml),
  which builds the Docker image for each CUDA version and pushes it to
  [ghcr.io/koszullab/instagraal](https://ghcr.io/koszullab/instagraal) as
  `X.Y.Z-cuda<cuda>` and `latest-cuda<cuda>`.

The release commit on `master` deploys the site with the new version.
