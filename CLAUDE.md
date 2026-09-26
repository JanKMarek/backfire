# CLAUDE.md - Quantitative Strategy Research Engine 

See README.md for: 
- Project Goals and Overview
- Application Domain description
- Architecture and Design
- CLI Workflows. 

See docs/WORKITEMS.md for active backlog and work items

## Tooling
- **Environment**: Python 3.13+, uv, jupyter lab, plotly charts/dashboards, pandas, LangChain and LLMs (Claude Fable/Opus). 
`uv.lock` is committed — `uv sync` reproduces the exact environment. Notebook and plotting
packages (jupyterlab, plotly, matplotlib) live in the `dev` dependency group; the `backfire`
package itself only needs pandas/numpy/pyyaml.
Use skill download-market-data to download OHLC data. Do NOT use yfinance. 
- **Setup**: `uv sync`
- **Issue tracking**: Github