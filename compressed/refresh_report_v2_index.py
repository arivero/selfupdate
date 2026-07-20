"""Refresh completion-ordered symlinks to individual report-v2 PDFs."""

from report_v2 import REPORT_INDEX, refresh_report_index


if __name__ == "__main__":
    count = refresh_report_index()
    print(f"indexed {count} reports under {REPORT_INDEX}")
