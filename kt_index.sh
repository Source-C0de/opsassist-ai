python -m pip install -r requriement.txt
PYTHONPATH=src python src/ingest.py run --dry-run
PYTHONPATH=src python src/index.py
PYTHONPATH=src python src/ingest.py upload