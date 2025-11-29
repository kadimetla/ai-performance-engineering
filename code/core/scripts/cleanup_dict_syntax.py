#!/usr/bin/env python3
"""Script to fix dict([...]) anti-pattern across the codebase.

Usage:
    python core/scripts/cleanup_dict_syntax.py [--dry-run]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


def fix_dict_pattern(content: str) -> tuple[str, int]:
    """Convert dict([...]) to {...} syntax.
    
    Returns:
        Tuple of (fixed_content, num_fixes)
    """
    num_fixes = 0
    
    # Pattern to match multi-line dict([...]) with any content inside
    # This handles return dict([...]) patterns
    pattern = re.compile(
        r'(return\s+)dict\(\[\s*\n((?:[^\]]*\n)*?)\s*\]\)',
        re.MULTILINE
    )
    
    def replace_dict(match: re.Match) -> str:
        nonlocal num_fixes
        num_fixes += 1
        prefix = match.group(1)  # 'return ' or 'return\n'
        entries_block = match.group(2)
        
        # Parse entries: ("key", value), with optional comment
        entry_pattern = re.compile(r'\s*\((["\'][^"\']+["\'])\s*,\s*([^)]+)\)\s*,?(\s*#.*)?')
        
        items = []
        for line in entries_block.split('\n'):
            line = line.strip()
            if not line:
                continue
            m = entry_pattern.match(line)
            if m:
                key = m.group(1)
                value = m.group(2).strip().rstrip(',')
                comment = m.group(3) or ''
                items.append(f'            {key}: {value},{comment}')
        
        if not items:
            return match.group(0)  # No change if parsing failed
        
        return prefix + '{\n' + '\n'.join(items) + '\n        }'
    
    content = pattern.sub(replace_dict, content)
    
    # Also handle single-line patterns
    single_pattern = re.compile(
        r'dict\(\[\s*\((["\'][^"\']+["\'])\s*,\s*([^)]+)\)\s*,?\s*\]\)'
    )
    
    def single_replace(match: re.Match) -> str:
        nonlocal num_fixes
        num_fixes += 1
        key = match.group(1)
        value = match.group(2).strip().rstrip(',')
        return f'{{{key}: {value}}}'
    
    content = single_pattern.sub(single_replace, content)
    
    return content, num_fixes


def process_file(filepath: Path, dry_run: bool = False) -> int:
    """Process a single file and return number of fixes."""
    try:
        content = filepath.read_text()
    except Exception as e:
        print(f"  Error reading {filepath}: {e}")
        return 0
    
    if 'dict([' not in content:
        return 0
    
    fixed_content, num_fixes = fix_dict_pattern(content)
    
    if num_fixes > 0:
        if dry_run:
            print(f"  Would fix {num_fixes} pattern(s) in {filepath}")
        else:
            filepath.write_text(fixed_content)
            print(f"  Fixed {num_fixes} pattern(s) in {filepath}")
    
    return num_fixes


def main() -> None:
    dry_run = '--dry-run' in sys.argv
    
    repo_root = Path(__file__).parent.parent
    
    # Find all Python files in chapter and lab directories
    patterns = [
        'ch*/baseline_*.py',
        'ch*/optimized_*.py',
        'ch*/**/baseline_*.py',
        'ch*/**/optimized_*.py',
        'labs/**/*.py',
        'core/benchmark/*.py',
        'profiling/*.py',
        'core/**/*.py',
        'optimization/*.py',
        'analysis/*.py',
        'core/scripts/**/*.py',
    ]
    
    total_fixes = 0
    files_fixed = 0
    
    for pattern in patterns:
        for filepath in repo_root.glob(pattern):
            if filepath.is_file():
                fixes = process_file(filepath, dry_run)
                if fixes > 0:
                    total_fixes += fixes
                    files_fixed += 1
    
    mode = "Would fix" if dry_run else "Fixed"
    print(f"\n{mode} {total_fixes} dict([...]) patterns in {files_fixed} files")
    
    if dry_run:
        print("\nRun without --dry-run to apply changes")


if __name__ == '__main__':
    main()
