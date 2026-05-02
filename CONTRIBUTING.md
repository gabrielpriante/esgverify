# Contributing to ESGVerify

Thank you for your interest in contributing. ESGVerify is an open-source
tool built for sustainability professionals, and community contributions
are what will make it genuinely useful.

---

## Before you start

- Check the [open issues](../../issues) to see what's already being worked on
- For significant changes, open an issue first to discuss the approach
- Keep pull requests focused — one feature or fix per PR

---

## Development setup

Follow the [Quickstart in the README](README.md#quickstart) to get the
backend and frontend running locally.

---

## Code standards

### Python (backend)

- **Formatter**: `black` with default settings
- **Linter**: `ruff`
- **Types**: All public functions must have type annotations
- **Docstrings**: Google-style docstrings on all modules, classes, and public functions
- **Tests**: New features require unit tests in `tests/unit/`

Run before committing:
```bash
black backend/
ruff check backend/
pytest tests/
```

### TypeScript / React (frontend)

- **Formatter**: Prettier with default settings
- **Linter**: ESLint
- **Types**: No `any` — use explicit types or generics

Run before committing:
```bash
npm run lint
npm run type-check
```

---

## Pull request checklist

- [ ] Code follows the style guide above
- [ ] Tests pass locally
- [ ] New behavior is covered by a test
- [ ] Docstrings updated if public API changed
- [ ] README or docs updated if needed

---

## Areas that need help

- Additional document format support (HTML, XLSX earnings calls)
- ESG framework schema files (TCFD, SASB, EU Taxonomy)
- Sample ESG documents for the `data/samples/` directory
- Frontend design improvements
- Non-English document support

---

## Code of conduct

Be constructive, assume good intent, and keep feedback focused on the work.
