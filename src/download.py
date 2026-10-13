"""Download the MIT-BIH Arrhythmia Database from PhysioNet."""

import argparse

from src import config
from src.utils import ensure_dirs, get_logger

log = get_logger("download")


def missing_records(raw_dir=config.RAW_DIR, records=config.ALL_RECORDS):
    return [r for r in records if not all((raw_dir / f"{r}{ext}").exists() for ext in (".dat", ".hea", ".atr"))]


def download(force=False):
    import wfdb

    ensure_dirs()
    todo = config.ALL_RECORDS if force else missing_records()
    if not todo:
        log.info("All %d records already in %s", len(config.ALL_RECORDS), config.RAW_DIR)
        return
    log.info("Downloading %d records -> %s", len(todo), config.RAW_DIR)
    wfdb.dl_database(config.PHYSIONET_DB, dl_dir=str(config.RAW_DIR), records=todo)
    if missing := missing_records():
        raise RuntimeError(f"Download incomplete, still missing: {missing}")
    log.info("Done (paced records %s are dropped later)", ", ".join(config.PACED_RECORDS))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download everything")
    download(parser.parse_args().force)
