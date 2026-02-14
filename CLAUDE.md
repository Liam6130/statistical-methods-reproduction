# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a statistical methods reproduction project - a repository for reproducing and implementing various statistical methods from academic papers. The project is designed to host multiple independent statistical method implementations as sub-projects.

## Project Structure

```
My_project/
├── main/                          # Main directory for all sub-projects
│   ├── project_template/          # Template for new sub-projects
│   │   ├── src/                  # Source code (Python3 primary)
│   │   ├── scripts/              # Utility scripts
│   │   ├── simulation/           # Simulation code and data
│   │   ├── data/                 # Input/data files
│   │   ├── results/             # Output results
│   │   ├── docs/                # Documentation (not uploaded by default)
│   │   ├── figures/              # Generated figures
│   │   ├── tests/               # Unit tests
│   │   └── README.md            # Project-specific README
│   └── README.md                 # Main README
├── .github/
│   └── workflows/                # CI/CD workflows (optional)
├── .gitignore
└── CLAUDE.md
```

## Technical Stack

- **Primary**: Python 3 (statistical methods implementation)
- **Secondary**: R, MATLAB, JavaScript
- **Package Management**: uv/pip for Python, renv for R
- **Testing**: pytest for Python

## Development Workflow

1. **Creating new sub-project**: Copy `project_template` to a new directory under `main/`
2. **Adding new method**: Create sub-directory under `main/` with descriptive name
3. **Code submission**: Via pull requests to main branch

## Common Commands

```bash
# Python development
uv pip install -r requirements.txt    # Install dependencies
pytest                                # Run tests
python -m pytest tests/               # Run specific test directory

# Project template structure check
ls -la main/project_template/
```

## Important Notes

- All source code goes into `src/` or `scripts/`
- Simulation data in `simulation/`, input data in `data/`
- Generated outputs in `results/` and `figures/`
- Documentation in `docs/` - NOT uploaded to GitHub by default
- Only upload publisher-copyrighted literature if explicitly permitted
- README.md is the only markdown file uploaded by default
