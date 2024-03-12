# Contributing

## Contributor License Agreement

We are grateful to the contributors who help evolve LangServe and dedicate their time to the project. As the primary sponsor of LangServe, LangChain, Inc. aims to build products in the open that benefit thousands of developers while allowing us to build a sustainable business. For all code contributions to LangServe, we ask that contributors complete and sign a Contributor License Agreement (“CLA”). The agreement between contributors and the project is explicit, so LangServe users can be confident in the legal status of the source code and their right to use it.The CLA does not change the terms of the underlying license, LangServe License, used by our software.

Before you can contribute to LangServe, a bot will comment on the PR asking you to agree to the CLA if you haven't already. Agreeing to the CLA is required before code can be merged and only needs to happen on the first contribution to the project. All subsequent contributions will fall under the same CLA.

## 🗺️ Guidelines

### Dependency Management: Poetry and other env/dependency managers

This project uses [Poetry](https://python-poetry.org/) v1.6.1+ as a dependency manager.

### Local Development Dependencies

Install langserve development requirements (for running langchain, running examples, linting, formatting, tests, and coverage):

```sh
poetry install --with test,dev
```

Then verify that tests pass:

```sh
make test
```

### Formatting and Linting

Run these locally before submitting a PR; the CI system will check also.

#### Code Formatting

Formatting for this project is done via a combination of [Black](https://black.readthedocs.io/en/stable/) and [ruff](https://docs.astral.sh/ruff/rules/).

To run formatting for this project:

```sh
make format
```

#### Linting

Linting for this project is done via a combination of [Black](https://black.readthedocs.io/en/stable/), [ruff](https://docs.astral.sh/ruff/rules/), and [mypy](http://mypy-lang.org/).

To run linting for this project:

```sh
make lint
```

## Frontend Playground Development

Here are a few tips to keep in mind when developing the LangServe playgrounds:

### Setup

Switch directories to `langserve/playground` or `langserve/chat_playground`, then run `yarn` to install required
dependencies. `yarn dev` will start the playground at `http://localhost:5173/____LANGSERVE_BASE_URL/` in dev mode.

You can run one of the chains in the `examples/` repo using `poetry run python path/to/file.py`.

### Setting CORS

You may need to add the following to an example route when developing the playground in dev mode to handle CORS:

```python
from fastapi.middleware.cors import CORSMiddleware

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)
```
