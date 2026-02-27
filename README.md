## Setup
==================

uv init

uv venv .venv

<!-- uv python pin 3.13 -->

source .venv/bin/activate

uv run uvicorn main:app --reload

http://127.0.0.1:8000


## Quick reference
==================
uv venv .venv
source .venv/bin/activate
uv sync

source .venv/bin/activate
uv run uvicorn main:app --reload
