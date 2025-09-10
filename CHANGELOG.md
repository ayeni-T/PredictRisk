# Changelog

All notable changes to this project will be documented in this file.

## [2.0.0] - 2025-09-10
### Added
- Safety-first **Overall Recommendation** with clear triage levels (Routine / Urgent / Emergency).
- Always-visible **Risk % + category + 95% CI**.
- **Condition-aware Clinical Guidance** generation.
- **PDF report** with logo header and **footer-only** disclaimer + © Taiwo Michael Ayeni.
- Streamlit theme & red primary action button.
- Script-relative paths for artifacts and logo.

### Changed
- Switched runtime inference to **Laplace** artifacts (tiny μ, Σ) for speed/size.
- Inputs start blank with placeholders; validations added.

### Removed
- Lab test requirements; kept **non-lab** scope only.
