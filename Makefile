.PHONY: serve test index

serve:
	@uvicorn rogue_copilot.server.app:app

test:
	@pytest

index:
	@python scripts/build_index.py
