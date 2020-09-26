from pathlib import Path

def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


