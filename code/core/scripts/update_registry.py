#!/usr/bin/env python3
"""Update file registry and concept mapping.

Wrapper script to regenerate both the file registry and concept mapping.
This should be run before coverage checks to ensure all files are registered.

Usage:
    python3 core/scripts/update_registry.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent


def main():
    """Main function."""
    print("=" * 80)
    print("UPDATING REGISTRY AND MAPPING")
    print("=" * 80)
    print()
    
    # Update file registry
    print("1. Updating file registry...")
    result = subprocess.run(
        [sys.executable, "core/scripts/generate_file_registry.py"],
        cwd=repo_root,
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print("[OK] File registry updated")
    else:
        print(f"ERROR: File registry update failed: {result.stderr}")
        return 1
    
    print()
    
    # Update concept mapping
    print("2. Updating concept mapping...")
    result = subprocess.run(
        [sys.executable, "core/scripts/generate_concept_mapping.py", "--preserve-existing"],
        cwd=repo_root,
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print("[OK] Concept mapping updated")
    else:
        print(f"ERROR: Concept mapping update failed: {result.stderr}")
        return 1
    
    print()
    print("=" * 80)
    print("[OK] Registry and mapping updated successfully!")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
