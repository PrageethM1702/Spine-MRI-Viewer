import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import get_subjects_by_vendor


def main():
    for vendor in ("All", "GE", "Philips", "Siemens", "Other"):
        print(f"{vendor}: {len(get_subjects_by_vendor(vendor))}")


if __name__ == "__main__":
    main()
