# Quickstart — Getting Started with `vvkit`

`vvkit` turns solver verification into a declarative, reproducible, CI-runnable artifact.

## 1. Installation

```bash
uv add vvkit
```

## 2. Initialize a Study Configuration

Run `vv init` to create a scaffolding `vvcase.yaml`:

```bash
vv init
```

## 3. Run Verification Study

Execute the study and view results:

```bash
vv run --config-path vvcase.yaml
```
