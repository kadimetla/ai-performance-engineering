#!/usr/bin/env python3
"""
LLM Patch Applier - Extracts and applies code patches from LLM analysis.

This module parses LLM-generated optimization suggestions, extracts code patches,
validates them with AST parsing, and applies them to create new optimized variants.

Supports two patching strategies:
1. AST-based (default): Replaces complete functions by name using AST manipulation
2. Fuzzy: Uses text-based fuzzy matching to find and replace code blocks

Usage:
    from core.analysis.llm_patch_applier import LLMPatchApplier
    
    applier = LLMPatchApplier(strategy="ast")
    patches = applier.extract_patches(llm_analysis_result)
    new_files = applier.apply_patches(patches, original_file, output_dir)
"""

from __future__ import annotations

import ast
import json
import importlib.util
import re
import sys
import tempfile
import textwrap
import difflib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from core.utils.logger import get_logger

logger = get_logger(__name__)


class PatchStrategy(str, Enum):
    """Available patching strategies."""
    AST = "ast"       # AST-based function replacement (more reliable)
    FUZZY = "fuzzy"   # Fuzzy text matching (legacy)


@dataclass
class FunctionReplacement:
    """Represents a complete function replacement extracted from LLM response."""
    function_name: str           # e.g., "benchmark_fn" or "ClassName.method_name"
    class_name: Optional[str]    # e.g., "OptimizedBenchmark" if method
    new_code: str                # Complete function/method code
    language: str = "python"     # python, cuda
    description: str = ""        # Optional description
    confidence: float = 0.9      # Higher confidence for structured replacements
    
    @property
    def is_method(self) -> bool:
        """Check if this is a class method replacement."""
        return self.class_name is not None
    
    @property
    def short_name(self) -> str:
        """Get just the function/method name without class."""
        return self.function_name.split('.')[-1] if '.' in self.function_name else self.function_name


@dataclass
class StructuredPatch:
    """Represents a complete structured patch from JSON LLM response."""
    variant_name: str                    # e.g., "stream_parallel_fp16"
    description: str                     # Human-readable description
    expected_speedup: str                # e.g., "1.3x"
    new_imports: List[str] = field(default_factory=list)  # New import statements
    init_additions: List[str] = field(default_factory=list)  # Lines to add to __init__
    method_replacements: List[Dict[str, str]] = field(default_factory=list)  # List of {class_name, method_name, complete_code}
    new_methods: List[Dict[str, str]] = field(default_factory=list)  # New helper methods to add
    full_class_replacement: Optional[Dict[str, str]] = None  # Optional full class replacement
    confidence: float = 0.95
    
    @property
    def safe_variant_name(self) -> str:
        """Get a filesystem-safe variant name."""
        # Clean up the name to be a valid identifier
        name = re.sub(r'[^a-zA-Z0-9_]', '_', self.variant_name)
        name = re.sub(r'_+', '_', name)  # Remove consecutive underscores
        name = name.strip('_').lower()
        return name if name else "optimized"
    
    def get_all_provided_methods(self) -> Set[str]:
        """Get all method names provided by this patch."""
        methods = set()
        for m in self.method_replacements:
            methods.add(m.get("method_name", ""))
        for m in self.new_methods:
            methods.add(m.get("method_name", ""))
        return methods
    
    @property
    def is_full_class_replacement(self) -> bool:
        """Check if this patch replaces an entire class."""
        return self.full_class_replacement is not None


@dataclass
class CodePatch:
    """Represents a code patch extracted from LLM response (legacy fuzzy format)."""
    description: str
    before_code: str
    after_code: str
    language: str = "python"
    confidence: float = 0.5
    patch_type: str = "replacement"
    
    def is_valid(self) -> bool:
        """Check if patch seems valid."""
        has_before = len(self.before_code.strip()) > 5
        has_after = len(self.after_code.strip()) > 5
        is_different = self.before_code.strip() != self.after_code.strip()
        return has_before and has_after and is_different


@dataclass
class PatchApplication:
    """Result of applying a patch."""
    success: bool
    original_file: Path
    patched_file: Optional[Path] = None
    patch: Optional[Union[FunctionReplacement, CodePatch]] = None
    error: Optional[str] = None
    diff_preview: str = ""
    validation_errors: List[str] = field(default_factory=list)
    strategy_used: str = "unknown"


class LLMPatchApplier:
    """Extracts and applies code patches from LLM analysis responses."""
    
    def __init__(
        self, 
        strategy: Union[str, PatchStrategy] = PatchStrategy.AST,
        dry_run: bool = False, 
        validate_syntax: bool = True
    ):
        """
        Initialize the patch applier.
        
        Args:
            strategy: Patching strategy - "ast" (default) or "fuzzy"
            dry_run: If True, don't actually write files
            validate_syntax: If True, validate Python syntax before saving
        """
        self.strategy = PatchStrategy(strategy) if isinstance(strategy, str) else strategy
        self.dry_run = dry_run
        self.validate_syntax = validate_syntax
    
    def extract_patches(self, llm_response: str) -> List[Union[StructuredPatch, FunctionReplacement, CodePatch]]:
        """
        Extract patches from LLM response text.
        
        For AST strategy:
        1. First tries to parse JSON structured patches
        2. Falls back to REPLACE_FUNCTION/REPLACE_METHOD blocks
        3. Finally tries legacy patterns
        
        For fuzzy strategy, uses legacy patterns.
        
        Args:
            llm_response: The raw LLM response text
            
        Returns:
            List of extracted patch objects
        """
        patches = []
        
        if self.strategy == PatchStrategy.AST:
            # Primary: Try to parse JSON structured patches
            structured = self._extract_json_patches(llm_response)
            if structured:
                patches.extend(structured)
                logger.info(f"Extracted {len(structured)} structured patches from JSON")
            else:
                # Fallback 1: Try REPLACE_FUNCTION/REPLACE_METHOD blocks
                replacements = self._extract_function_replacements(llm_response)
                if replacements:
                    patches.extend(replacements)
                    logger.info(f"Extracted {len(replacements)} function replacements")
                else:
                    # Fallback 2: Try legacy patterns
                    logger.info("No structured patches found, trying legacy extraction")
                    patches.extend(self._extract_legacy_patches(llm_response))
        else:
            # Fuzzy strategy uses legacy extraction
            patches.extend(self._extract_legacy_patches(llm_response))
        
        logger.info(f"Extracted {len(patches)} patches using {self.strategy.value} strategy")
        return patches
    
    def _extract_json_patches(self, text: str) -> List[StructuredPatch]:
        """Extract patches from JSON structured response."""
        patches = []
        
        # Try to find JSON with code_patches in the response
        # Handle cases where JSON is in a code block that might not be closed
        try:
            # Look for code_patches or patches array in the text
            code_patches_match = re.search(r'"code_patches"\s*:\s*\[', text)
            if not code_patches_match:
                # Also try "patches" as some LLMs use that key
                code_patches_match = re.search(r'"patches"\s*:\s*\[', text)
            if not code_patches_match:
                return []
            
            # Find the start of the enclosing JSON object
            start = text.rfind('{', 0, code_patches_match.start())
            if start == -1:
                start = text.find('{')
                if start == -1:
                    return []
            
            # Try to find proper JSON end, handle incomplete JSON
            json_str = text[start:]
            
            # Try parsing as-is first
            data = None
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                # JSON might be incomplete - try to extract just code_patches array
                pass
            
            if data is None:
                # Extract just the code_patches or patches array
                array_start = json_str.find('"code_patches"')
                if array_start == -1:
                    array_start = json_str.find('"patches"')
                if array_start == -1:
                    return []
                
                # Find the array start
                bracket_start = json_str.find('[', array_start)
                if bracket_start == -1:
                    return []
                
                # Find matching bracket, handling nested objects
                depth = 0
                bracket_end = bracket_start
                in_string = False
                i = bracket_start
                
                while i < len(json_str):
                    c = json_str[i]
                    
                    if in_string:
                        if c == '\\' and i + 1 < len(json_str):
                            # Skip escaped character
                            i += 2
                            continue
                        elif c == '"':
                            in_string = False
                    else:
                        if c == '"':
                            in_string = True
                        elif c == '[':
                            depth += 1
                        elif c == ']':
                            depth -= 1
                            if depth == 0:
                                bracket_end = i + 1
                                break
                    i += 1
                
                if bracket_end > bracket_start:
                    array_str = json_str[bracket_start:bracket_end]
                    try:
                        raw_patches = json.loads(array_str)
                        data = {"code_patches": raw_patches}
                    except json.JSONDecodeError as e:
                        logger.debug(f"Array parse failed: {e}")
                        # Try to extract individual complete objects from truncated array
                        raw_patches = self._extract_complete_objects_from_truncated(json_str[bracket_start:])
                        if raw_patches:
                            data = {"code_patches": raw_patches}
                            logger.info(f"Recovered {len(raw_patches)} complete patches from truncated JSON")
                        else:
                            return []
                else:
                    # Array was truncated (never closed) - try to recover complete objects
                    logger.debug("Array appears truncated, attempting recovery")
                    raw_patches = self._extract_complete_objects_from_truncated(json_str[bracket_start:])
                    if raw_patches:
                        data = {"code_patches": raw_patches}
                        logger.info(f"Recovered {len(raw_patches)} complete patches from truncated JSON")
                    else:
                        return []
            
            if data is None:
                return []
            
            raw_patches = data.get("code_patches") or data.get("patches", [])
            
            for p in raw_patches:
                if not isinstance(p, dict):
                    continue
                patch = StructuredPatch(
                    variant_name=p.get("variant_name", "optimized"),
                    description=p.get("description", ""),
                    expected_speedup=p.get("expected_speedup", "unknown"),
                    new_imports=p.get("new_imports", []),
                    init_additions=p.get("init_changes", {}).get("add_attributes", []) if isinstance(p.get("init_changes"), dict) else [],
                    method_replacements=p.get("method_replacements", []),
                    new_methods=p.get("new_methods", []),
                    full_class_replacement=p.get("full_class_replacement"),
                )
                patches.append(patch)
                logger.info(f"Extracted structured patch: {patch.variant_name} ({patch.expected_speedup})")
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as e:
            logger.debug(f"JSON parsing failed: {e}")
        
        return patches
    
    def _extract_complete_objects_from_truncated(self, text: str) -> List[Dict[str, Any]]:
        """Extract complete JSON objects from a truncated array.
        
        When LLM output is truncated mid-JSON, this tries to salvage
        any complete patch objects from the beginning of the array.
        """
        objects = []
        
        if not text.startswith('['):
            return objects
        
        # Find complete objects in the array
        i = 1  # Skip opening [
        while i < len(text):
            # Skip whitespace and commas
            while i < len(text) and text[i] in ' \n\t\r,':
                i += 1
            
            if i >= len(text) or text[i] != '{':
                break
            
            # Find matching closing brace
            obj_start = i
            depth = 0
            in_string = False
            
            while i < len(text):
                c = text[i]
                
                if in_string:
                    if c == '\\' and i + 1 < len(text):
                        i += 2
                        continue
                    elif c == '"':
                        in_string = False
                else:
                    if c == '"':
                        in_string = True
                    elif c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            # Found complete object
                            obj_str = text[obj_start:i+1]
                            try:
                                obj = json.loads(obj_str)
                                # Validate it has required fields for a patch
                                if 'variant_name' in obj or 'method_replacements' in obj:
                                    objects.append(obj)
                                    logger.debug(f"Recovered object: {obj.get('variant_name', 'unnamed')}")
                            except json.JSONDecodeError:
                                pass
                            i += 1
                            break
                i += 1
            else:
                # Didn't find closing brace, array is truncated here
                break
        
        return objects
    
    def _extract_function_replacements(self, text: str) -> List[FunctionReplacement]:
        """Extract structured REPLACE_FUNCTION/REPLACE_METHOD blocks."""
        replacements = []
        
        # Pattern 1: ### REPLACE_FUNCTION: name or ### REPLACE_METHOD: Class.name followed by code block
        # Also handles the case where it's inside an existing code block
        patterns = [
            # Standard format: ### REPLACE_METHOD: Name\n```python\ndef ...```
            r'###\s*REPLACE_(?:FUNCTION|METHOD):\s*(\S+)\s*\n```(?:python|cuda)?\n(.*?)```',
            # Inside code block: ```python\n### REPLACE_METHOD: Name\ndef ...```
            r'```(?:python|cuda)?\n###\s*REPLACE_(?:FUNCTION|METHOD):\s*(\S+)\s*\n((?:def|async def|class)[^`]+?)```',
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.DOTALL | re.IGNORECASE):
                full_name = match.group(1).strip()
                code = match.group(2).strip()
                
                # Parse class.method format
                if '.' in full_name:
                    parts = full_name.split('.', 1)
                    class_name = parts[0]
                    func_name = parts[1]
                else:
                    class_name = None
                    func_name = full_name
                
                # Validate the code is syntactically correct
                try:
                    # Try to parse as a function/method
                    ast.parse(code)
                    
                    replacements.append(FunctionReplacement(
                        function_name=func_name,
                        class_name=class_name,
                        new_code=code,
                        language="python",
                        description=f"Replace {full_name}",
                        confidence=0.95,
                    ))
                    logger.debug(f"Extracted function replacement: {full_name}")
                except SyntaxError as e:
                    logger.warning(f"Skipping invalid function replacement {full_name}: {e}")
        
        return replacements
    
    def _extract_legacy_patches(self, text: str) -> List[CodePatch]:
        """Extract patches using legacy patterns (before/after, diffs)."""
        patches = []
        
        # Pattern 1: Before/After blocks
        patches.extend(self._extract_before_after_blocks(text))
        
        # Pattern 2: Diff format blocks
        patches.extend(self._extract_diff_blocks(text))
        
        # Pattern 3: Inline suggestions
        patches.extend(self._extract_inline_suggestions(text))
        
        # Pattern 4: Replace X with Y patterns
        patches.extend(self._extract_replace_patterns(text))
        
        # Deduplicate
        seen = set()
        unique = []
        for patch in patches:
            if patch.is_valid():
                key = (
                    self._normalize_code(patch.before_code),
                    self._normalize_code(patch.after_code)
                )
                if key not in seen:
                    seen.add(key)
                    unique.append(patch)
        
        return unique
    
    def _normalize_code(self, code: str) -> str:
        """Normalize code for comparison."""
        lines = code.strip().split('\n')
        if lines:
            min_indent = min((len(line) - len(line.lstrip()) for line in lines if line.strip()), default=0)
            lines = [line[min_indent:] if len(line) >= min_indent else line for line in lines]
        return '\n'.join(lines).strip()
    
    def _extract_before_after_blocks(self, text: str) -> List[CodePatch]:
        """Extract Before/After style patches."""
        patches = []
        patterns = [
            r'\*\*Before[:\s]*\*\*\s*```(\w*)\n(.*?)```\s*\*\*After[:\s]*\*\*\s*```(\w*)\n(.*?)```',
            r'Before[:\s]+```(\w*)\n(.*?)```\s*After[:\s]+```(\w*)\n(.*?)```',
            r'###\s*Before[:\s]*\n```(\w*)\n(.*?)```\s*###\s*After[:\s]*\n```(\w*)\n(.*?)```',
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.DOTALL | re.IGNORECASE):
                lang1, before, lang2, after = match.groups()
                lang = lang1 or lang2 or "python"
                if lang in ("cpp", "c++", "cuda", "cu"):
                    lang = "cuda"
                
                patches.append(CodePatch(
                    description="Before/After block",
                    before_code=before.strip(),
                    after_code=after.strip(),
                    language=lang,
                    confidence=0.9,
                ))
        
        return patches
    
    def _extract_diff_blocks(self, text: str) -> List[CodePatch]:
        """Extract diff-style patches."""
        patches = []
        diff_pattern = r'```diff\n(.*?)```'
        diff_blocks = list(re.finditer(diff_pattern, text, re.DOTALL))
        
        i = 0
        while i < len(diff_blocks):
            diff_content = diff_blocks[i].group(1)
            
            before_lines = []
            after_lines = []
            only_minus = True
            only_plus = True
            
            for line in diff_content.split('\n'):
                stripped = line.rstrip()
                if stripped.startswith('---') or stripped.startswith('+++') or stripped.startswith('@@'):
                    continue
                elif stripped.startswith('-') and len(stripped) > 1:
                    before_lines.append(stripped[1:])
                    only_plus = False
                elif stripped.startswith('+') and len(stripped) > 1:
                    after_lines.append(stripped[1:])
                    only_minus = False
                elif stripped.startswith(' '):
                    before_lines.append(stripped[1:])
                    after_lines.append(stripped[1:])
                    only_minus = False
                    only_plus = False
            
            # Handle paired diff blocks
            if only_minus and before_lines and i + 1 < len(diff_blocks):
                next_content = diff_blocks[i + 1].group(1)
                next_after = []
                next_only_plus = True
                
                for line in next_content.split('\n'):
                    stripped = line.rstrip()
                    if stripped.startswith('+') and len(stripped) > 1:
                        next_after.append(stripped[1:])
                    elif stripped.startswith('-'):
                        next_only_plus = False
                
                if next_only_plus and next_after:
                    patches.append(CodePatch(
                        description="Paired diff blocks",
                        before_code='\n'.join(before_lines),
                        after_code='\n'.join(next_after),
                        language="python",
                        confidence=0.8,
                    ))
                    i += 2
                    continue
            
            if before_lines and after_lines:
                patches.append(CodePatch(
                    description="Diff block",
                    before_code='\n'.join(before_lines),
                    after_code='\n'.join(after_lines),
                    language="python",
                    confidence=0.85,
                ))
            
            i += 1
        
        return patches
    
    def _extract_inline_suggestions(self, text: str) -> List[CodePatch]:
        """Extract suggestions from consecutive code blocks."""
        patches = []
        code_blocks = list(re.finditer(r'```(\w*)\n(.*?)```', text, re.DOTALL))
        
        for i in range(len(code_blocks) - 1):
            block1 = code_blocks[i]
            block2 = code_blocks[i + 1]
            
            if block2.start() - block1.end() < 300:
                lang1, code1 = block1.groups()
                lang2, code2 = block2.groups()
                
                if lang1 == 'diff' or lang2 == 'diff':
                    continue
                
                context = text[block1.end():block2.start()].lower()
                if any(word in context for word in ['after', 'becomes', 'change to', 'optimized', 'improved', 'should be', 'replace with']):
                    lang = lang1 or lang2 or "python"
                    if lang in ("cpp", "c++", "cuda", "cu"):
                        lang = "cuda"
                    
                    patches.append(CodePatch(
                        description="Sequential code blocks",
                        before_code=code1.strip(),
                        after_code=code2.strip(),
                        language=lang,
                        confidence=0.7,
                    ))
        
        return patches
    
    def _extract_replace_patterns(self, text: str) -> List[CodePatch]:
        """Extract 'Replace X with Y' patterns."""
        patches = []
        pattern = r'[Rr]eplace\s+`([^`]+)`\s+with\s+`([^`]+)`'
        for match in re.finditer(pattern, text):
            before, after = match.groups()
            if len(before.strip()) > 3 and len(after.strip()) > 3:
                patches.append(CodePatch(
                    description="Replace pattern",
                    before_code=before.strip(),
                    after_code=after.strip(),
                    language="python",
                    confidence=0.6,
                ))
        return patches
    
    def apply_patches(
        self,
        patches: List[Union[StructuredPatch, FunctionReplacement, CodePatch]],
        original_file: Path,
        output_dir: Path,
        variant_prefix: str = "llm_optimized",
    ) -> List[PatchApplication]:
        """
        Apply patches to create new file variants.
        
        Args:
            patches: List of patches to apply
            original_file: The original source file
            output_dir: Directory to write patched files
            variant_prefix: Prefix for new file variants
            
        Returns:
            List of PatchApplication results
        """
        results = []
        
        if not original_file.exists():
            logger.error(f"Original file not found: {original_file}")
            return results
        
        original_content = original_file.read_text()
        is_python = original_file.suffix == '.py'
        is_cuda = original_file.suffix in ('.cu', '.cuh')
        
        # Sort by confidence
        sorted_patches = sorted(patches, key=lambda p: p.confidence, reverse=True)
        
        for i, patch in enumerate(sorted_patches):
            if isinstance(patch, StructuredPatch):
                # Use meaningful variant name from the patch
                result = self._apply_structured_patch(
                    patch, original_content, original_file, output_dir
                )
            elif isinstance(patch, FunctionReplacement):
                result = self._apply_function_replacement(
                    patch, original_content, original_file, output_dir,
                    variant_prefix, variant_num=i + 1
                )
            else:
                result = self._apply_fuzzy_patch(
                    patch, original_content, original_file, output_dir,
                    variant_prefix, variant_num=i + 1,
                    validate_python=is_python, validate_cuda=is_cuda
                )
            results.append(result)
        
        return results
    
    def _apply_structured_patch(
        self,
        patch: StructuredPatch,
        original_content: str,
        original_file: Path,
        output_dir: Path,
    ) -> PatchApplication:
        """Apply a structured patch with __init__ changes, imports, and method replacements."""
        
        try:
            # Parse the original file
            tree = ast.parse(original_content)
        except SyntaxError as e:
            return PatchApplication(
                success=False,
                original_file=original_file,
                patch=patch,
                error=f"Original file has syntax error: {e}",
                strategy_used="structured",
            )
        
        new_content = original_content
        lines = new_content.split('\n')
        
        # Check if this is a full class replacement
        if patch.is_full_class_replacement and patch.full_class_replacement:
            return self._apply_full_class_replacement(
                patch, original_content, original_file, output_dir, tree
            )
        
        # 1. Add new imports at the top (after existing imports)
        if patch.new_imports:
            import_insert_line = 0
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)) and hasattr(node, 'lineno'):
                    import_insert_line = max(import_insert_line, node.lineno)
            if import_insert_line == 0 and tree.body:
                first_stmt = tree.body[0]
                if isinstance(first_stmt, ast.Expr) and isinstance(getattr(first_stmt, "value", None), ast.Constant):
                    if isinstance(first_stmt.value.value, str):
                        import_insert_line = first_stmt.end_lineno or first_stmt.lineno
            
            # Insert new imports after existing ones
            for imp in reversed(patch.new_imports):
                stripped = imp.strip()
                if not stripped:
                    continue
                lines.insert(import_insert_line, stripped)
            
            new_content = '\n'.join(lines)
            # Re-parse after modifying
            try:
                tree = ast.parse(new_content)
            except SyntaxError as e:
                return PatchApplication(
                    success=False,
                    original_file=original_file,
                    patch=patch,
                    error=f"Syntax error after adding imports: {e}",
                    strategy_used="structured",
                )
            lines = new_content.split('\n')
        
        # 2. Add __init__ attribute additions
        if patch.init_additions:
            # Find the target class from method_replacements
            target_class = None
            for m in patch.method_replacements:
                if m.get("class_name"):
                    target_class = m.get("class_name")
                    break
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Only modify the target class (or any class if not specified)
                    if target_class and node.name != target_class:
                        continue
                    
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                            # Find the end of __init__
                            init_end = item.end_lineno or item.lineno
                            
                            # Get indentation of __init__ body
                            indent = "        "  # Default 8 spaces
                            if item.body:
                                first_stmt = item.body[0]
                                if hasattr(first_stmt, 'lineno'):
                                    first_line = lines[first_stmt.lineno - 1]
                                    indent = ' ' * (len(first_line) - len(first_line.lstrip()))
                            
                            # Insert attribute additions at end of __init__
                            for attr in reversed(patch.init_additions):
                                lines.insert(init_end, indent + attr)
                            break
                    break  # Only modify first matching class
            
            new_content = '\n'.join(lines)
            # Re-parse
            try:
                tree = ast.parse(new_content)
            except SyntaxError as e:
                return PatchApplication(
                    success=False,
                    original_file=original_file,
                    patch=patch,
                    error=f"Syntax error after adding __init__ attributes: {e}",
                    strategy_used="structured",
                )
            lines = new_content.split('\n')
        
        # 3. Pre-validation: Check method references before applying
        validation_issues = self._validate_method_references(patch, original_content)
        if validation_issues:
            logger.warning(f"Method reference validation issues: {validation_issues}")
            # Continue anyway but note the issues
        
        # 4. Apply method replacements
        for method_info in patch.method_replacements:
            class_name = method_info.get("class_name", "")
            method_name = method_info.get("method_name", "")
            new_code = method_info.get("complete_code", "")
            
            if not method_name or not new_code:
                continue
            
            # Handle escaped newlines in JSON
            new_code = new_code.replace('\\n', '\n')
            
            # Find and replace the method
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and (not class_name or node.name == class_name):
                    for i, item in enumerate(node.body):
                        if isinstance(item, ast.FunctionDef) and item.name == method_name:
                            start_line = item.lineno
                            end_line = item.end_lineno or start_line
                            
                            # Get indentation
                            original_line = lines[start_line - 1]
                            indent = len(original_line) - len(original_line.lstrip())
                            
                            # Indent the new code
                            new_lines = new_code.split('\n')
                            indented_new = []
                            for j, line in enumerate(new_lines):
                                if line.strip():
                                    # Don't double-indent
                                    stripped = line.lstrip()
                                    line_indent = len(line) - len(stripped)
                                    if line_indent == 0:
                                        indented_new.append(' ' * indent + stripped)
                                    else:
                                        indented_new.append(' ' * indent + line)
                                else:
                                    indented_new.append('')
                            
                            # Replace
                            lines = lines[:start_line - 1] + indented_new + lines[end_line:]
                            new_content = '\n'.join(lines)
                            
                            # Re-parse for next replacement
                            try:
                                tree = ast.parse(new_content)
                            except SyntaxError as e:
                                return PatchApplication(
                                    success=False,
                                    original_file=original_file,
                                    patch=patch,
                                    error=f"Syntax error after replacing {method_name}: {e}",
                                    strategy_used="structured",
                                )
                            lines = new_content.split('\n')
                            break
        
        # 5. Add new helper methods
        if patch.new_methods:
            # Find the target class from method_replacements
            target_class = None
            for m in patch.method_replacements:
                if m.get("class_name"):
                    target_class = m.get("class_name")
                    break
            
            for method_info in patch.new_methods:
                class_name = method_info.get("class_name", target_class)
                method_name = method_info.get("method_name", "")
                new_code = method_info.get("complete_code", "")
                
                if not method_name or not new_code:
                    continue
                
                # Handle escaped newlines
                new_code = new_code.replace('\\n', '\n')
                
                # Find the class and add the method at the end
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef) and (not class_name or node.name == class_name):
                        # Find the end of the class
                        class_end = node.end_lineno or node.lineno
                        
                        # Get indentation (method level)
                        indent = "    "  # Default 4 spaces for method
                        if node.body:
                            first_item = node.body[0]
                            if hasattr(first_item, 'lineno') and first_item.lineno > 0:
                                first_line = lines[first_item.lineno - 1]
                                indent = ' ' * (len(first_line) - len(first_line.lstrip()))
                        
                        # Indent the new method
                        new_lines = new_code.split('\n')
                        indented_new = ['']  # Blank line before new method
                        for line in new_lines:
                            if line.strip():
                                stripped = line.lstrip()
                                line_indent = len(line) - len(stripped)
                                if line_indent == 0:
                                    indented_new.append(indent + stripped)
                                else:
                                    indented_new.append(indent + line)
                            else:
                                indented_new.append('')
                        
                        # Insert before the class end
                        lines = lines[:class_end] + indented_new + lines[class_end:]
                        new_content = '\n'.join(lines)
                        
                        # Re-parse
                        try:
                            tree = ast.parse(new_content)
                        except SyntaxError as e:
                            return PatchApplication(
                                success=False,
                                original_file=original_file,
                                patch=patch,
                                error=f"Syntax error after adding new method {method_name}: {e}",
                                strategy_used="structured",
                            )
                        lines = new_content.split('\n')
                        break
        
        # Validate final result
        validation_errors = self._validate_python_syntax(new_content)
        
        # Generate diff preview
        diff = list(difflib.unified_diff(
            original_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=str(original_file),
            tofile=f"{patch.safe_variant_name}",
            lineterm='',
        ))
        diff_preview = ''.join(diff[:100])
        
        # Create output file with meaningful name
        stem = original_file.stem
        suffix = original_file.suffix
        new_filename = f"{stem}_{patch.safe_variant_name}{suffix}"
        output_path = output_dir / new_filename
        
        # Write the file
        if self.dry_run:
            logger.info(f"[DRY RUN] Would create: {output_path}")
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path.write_text(new_content)
            logger.info(f"Created patched variant: {output_path}")
        
        # Import-time validation
        import_errors = []
        if not self.dry_run and not validation_errors and output_path.suffix == '.py':
            import_errors = self._validate_import(output_path)
        
        all_errors = validation_errors + import_errors
        
        return PatchApplication(
            success=True,
            original_file=original_file,
            patched_file=output_path,
            patch=patch,
            diff_preview=diff_preview,
            validation_errors=all_errors,
            strategy_used="structured",
        )
    
    def _validate_import(self, file_path: Path) -> List[str]:
        """Validate that a Python file can be imported without errors."""
        errors = []
        try:
            # Try to compile the file
            with open(file_path, 'r') as f:
                code = f.read()
            compile(code, str(file_path), 'exec')
        except SyntaxError as e:
            errors.append(f"Import validation failed (syntax): {e}")
        except Exception as e:
            errors.append(f"Import validation failed: {e}")
        
        return errors
    
    def _validate_method_references(self, patch: StructuredPatch, original_content: str) -> List[str]:
        """Validate that all method references in the patch can be resolved."""
        issues = []
        
        # Get existing methods from original code
        existing_methods = set()
        try:
            tree = ast.parse(original_content)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            existing_methods.add(item.name)
        except SyntaxError:
            return ["Could not parse original code"]
        
        # Get methods provided by the patch
        provided_methods = patch.get_all_provided_methods()
        all_available = existing_methods | provided_methods
        
        # Check each replacement for self.xxx() calls
        for method_info in patch.method_replacements + patch.new_methods:
            code = method_info.get("complete_code", "").replace('\\n', '\n')
            method_name = method_info.get("method_name", "unknown")
            
            # Find all self.method() calls
            method_call_pattern = r'self\.([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
            for match in re.finditer(method_call_pattern, code):
                called_method = match.group(1)
                # Skip common built-in/inherited methods
                if called_method in {'_synchronize', '_nvtx_range', 'to', 'half', 'eval', 
                                     'register_workload_metadata', 'cuda', 'cpu', 'zero_',
                                     'copy_', 'item', 'numpy', 'detach', 'clone', 'view',
                                     'reshape', 'unsqueeze', 'squeeze', 'expand', 'contiguous',
                                     'forward', '__init__', 'setup', 'benchmark_fn', 'teardown'}:
                    continue
                if called_method not in all_available:
                    issues.append(f"Method '{method_name}' calls self.{called_method}() which doesn't exist")
        
        # Check for self.xxx attribute access (not just method calls)
        for method_info in patch.method_replacements:
            code = method_info.get("complete_code", "").replace('\\n', '\n')
            method_name = method_info.get("method_name", "unknown")
            
            # Find self.attribute accesses (not followed by parenthesis)
            attr_pattern = r'self\.([a-zA-Z_][a-zA-Z0-9_]*)(?!\s*\()'
            referenced_attrs = set(re.findall(attr_pattern, code))
            
            # Get attributes initialized in __init__ from patch
            patch_attrs = set()
            for attr in patch.init_additions:
                match = re.match(r'self\.([a-zA-Z_][a-zA-Z0-9_]*)', attr)
                if match:
                    patch_attrs.add(match.group(1))
            
            # Get attributes from original __init__
            original_attrs = set()
            try:
                tree = ast.parse(original_content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                                for stmt in ast.walk(item):
                                    if isinstance(stmt, ast.Assign):
                                        for target in stmt.targets:
                                            if isinstance(target, ast.Attribute):
                                                if isinstance(target.value, ast.Name) and target.value.id == 'self':
                                                    original_attrs.add(target.attr)
            except SyntaxError:
                pass
            
            all_attrs = original_attrs | patch_attrs
            
            # Common/inherited attributes to skip
            skip_attrs = {'device', 'dtype', 'model', 'input_data', 'output', 'config',
                         'experts', 'router', 'num_experts', 'top_k', '_workload',
                         'batch_size', 'hidden_dim', 'num_requests', 'num_streams',
                         'streams', 'graphs', 'static_inputs', 'static_outputs',
                         'host_requests', 'host_outputs', 'device_inputs', 'device_outputs'}
            
            for attr in referenced_attrs:
                if attr not in all_attrs and attr not in skip_attrs and not attr.startswith('_'):
                    issues.append(f"Method '{method_name}' uses self.{attr} which may not be initialized")
        
        return issues
    
    def _apply_full_class_replacement(
        self,
        patch: StructuredPatch,
        original_content: str,
        original_file: Path,
        output_dir: Path,
        tree: ast.AST,
    ) -> PatchApplication:
        """Replace an entire class with a new implementation."""
        
        class_info = patch.full_class_replacement
        if not class_info:
            return PatchApplication(
                success=False,
                original_file=original_file,
                patch=patch,
                error="No class replacement info provided",
                strategy_used="structured",
            )
        
        class_name = class_info.get("class_name", "")
        new_class_code = class_info.get("complete_code", "")
        
        if not class_name or not new_class_code:
            return PatchApplication(
                success=False,
                original_file=original_file,
                patch=patch,
                error="Missing class_name or complete_code",
                strategy_used="structured",
            )
        
        # Handle escaped newlines
        new_class_code = new_class_code.replace('\\n', '\n')
        
        # Validate the new class code
        try:
            ast.parse(new_class_code)
        except SyntaxError as e:
            return PatchApplication(
                success=False,
                original_file=original_file,
                patch=patch,
                error=f"New class code has syntax error: {e}",
                strategy_used="structured",
            )
        
        # Find the class in the original tree
        lines = original_content.split('\n')
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                start_line = node.lineno
                end_line = node.end_lineno or start_line
                
                # Determine indentation of the original class
                original_line = lines[start_line - 1]
                indent = len(original_line) - len(original_line.lstrip())
                
                # Indent the new class code properly
                new_lines = []
                for line in new_class_code.split('\n'):
                    if line.strip():
                        # Apply indentation
                        stripped = line.lstrip()
                        line_indent = len(line) - len(stripped)
                        if line_indent == 0:
                            new_lines.append(' ' * indent + stripped)
                        else:
                            new_lines.append(' ' * indent + line)
                    else:
                        new_lines.append('')
                
                # Replace the class
                lines = lines[:start_line - 1] + new_lines + lines[end_line:]
                new_content = '\n'.join(lines)
                
                # Validate final result
                validation_errors = self._validate_python_syntax(new_content)
                
                # Generate diff preview
                diff = list(difflib.unified_diff(
                    original_content.splitlines(keepends=True),
                    new_content.splitlines(keepends=True),
                    fromfile=str(original_file),
                    tofile=f"{patch.safe_variant_name}",
                    lineterm='',
                ))
                diff_preview = ''.join(diff[:100])
                
                # Create output file
                stem = original_file.stem
                suffix = original_file.suffix
                new_filename = f"{stem}_{patch.safe_variant_name}{suffix}"
                output_path = output_dir / new_filename
                
                if not self.dry_run:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path.write_text(new_content)
                    logger.info(f"Created patched variant (full class replacement): {output_path}")
                
                return PatchApplication(
                    success=True,
                    original_file=original_file,
                    patched_file=output_path,
                    patch=patch,
                    diff_preview=diff_preview,
                    validation_errors=validation_errors,
                    strategy_used="structured",
                )
        
        return PatchApplication(
            success=False,
            original_file=original_file,
            patch=patch,
            error=f"Class '{class_name}' not found in original file",
            strategy_used="structured",
        )
    
    def _apply_function_replacement(
        self,
        replacement: FunctionReplacement,
        original_content: str,
        original_file: Path,
        output_dir: Path,
        variant_prefix: str,
        variant_num: int,
    ) -> PatchApplication:
        """Apply a function replacement using AST manipulation."""
        
        try:
            # Parse the original file
            tree = ast.parse(original_content)
        except SyntaxError as e:
            return PatchApplication(
                success=False,
                original_file=original_file,
                patch=replacement,
                error=f"Original file has syntax error: {e}",
                strategy_used="ast",
            )
        
        # Find the function/method to replace
        target_name = replacement.short_name
        target_class = replacement.class_name
        
        replacement_made = False
        new_content = original_content
        
        # Parse the replacement code to get the new function AST
        try:
            new_func_tree = ast.parse(replacement.new_code)
            if not new_func_tree.body:
                raise ValueError("Replacement code is empty")
            new_func = new_func_tree.body[0]
        except SyntaxError as e:
            return PatchApplication(
                success=False,
                original_file=original_file,
                patch=replacement,
                error=f"Replacement code has syntax error: {e}",
                strategy_used="ast",
            )
        
        # Find and replace the function
        for node in ast.walk(tree):
            if target_class:
                # Looking for a method in a class
                if isinstance(node, ast.ClassDef) and node.name == target_class:
                    for i, item in enumerate(node.body):
                        if isinstance(item, ast.FunctionDef) and item.name == target_name:
                            # Found the method - get its line range
                            start_line = item.lineno
                            end_line = item.end_lineno or start_line
                            
                            # Replace in the source
                            lines = original_content.split('\n')
                            
                            # Get the indentation of the original method
                            original_line = lines[start_line - 1]
                            indent = len(original_line) - len(original_line.lstrip())
                            
                            # Indent the new code
                            new_lines = replacement.new_code.split('\n')
                            indented_new = []
                            for j, line in enumerate(new_lines):
                                if line.strip():
                                    indented_new.append(' ' * indent + line)
                                else:
                                    indented_new.append('')
                            
                            # Replace
                            new_content = '\n'.join(
                                lines[:start_line - 1] +
                                indented_new +
                                lines[end_line:]
                            )
                            replacement_made = True
                            break
            else:
                # Looking for a top-level function
                if isinstance(node, ast.FunctionDef) and node.name == target_name:
                    start_line = node.lineno
                    end_line = node.end_lineno or start_line
                    
                    lines = original_content.split('\n')
                    new_content = '\n'.join(
                        lines[:start_line - 1] +
                        replacement.new_code.split('\n') +
                        lines[end_line:]
                    )
                    replacement_made = True
                    break
        
        if not replacement_made:
            return PatchApplication(
                success=False,
                original_file=original_file,
                patch=replacement,
                error=f"Could not find function/method '{replacement.function_name}' in file",
                strategy_used="ast",
            )
        
        # Validate the result
        validation_errors = []
        if self.validate_syntax:
            validation_errors = self._validate_python_syntax(new_content)
            if validation_errors:
                # Try to fix common issues
                fixed = self._try_fix_python(new_content, validation_errors)
                if fixed:
                    new_errors = self._validate_python_syntax(fixed)
                    if not new_errors:
                        new_content = fixed
                        validation_errors = []
        
        # Generate diff preview
        diff = list(difflib.unified_diff(
            original_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=str(original_file),
            tofile=f"{variant_prefix}_v{variant_num}",
            lineterm='',
        ))
        diff_preview = ''.join(diff[:100])
        
        # Create output file
        stem = original_file.stem
        suffix = original_file.suffix
        new_filename = f"{stem}_{variant_prefix}_v{variant_num}{suffix}"
        output_path = output_dir / new_filename
        
        if self.dry_run:
            logger.info(f"[DRY RUN] Would create: {output_path}")
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path.write_text(new_content)
            if validation_errors:
                logger.warning(f"Created patched file with validation warnings: {output_path}")
            else:
                logger.info(f"Created patched file: {output_path}")
        
        return PatchApplication(
            success=True,
            original_file=original_file,
            patched_file=output_path,
            patch=replacement,
            diff_preview=diff_preview,
            validation_errors=validation_errors,
            strategy_used="ast",
        )
    
    def _apply_fuzzy_patch(
        self,
        patch: CodePatch,
        original_content: str,
        original_file: Path,
        output_dir: Path,
        variant_prefix: str,
        variant_num: int,
        validate_python: bool = False,
        validate_cuda: bool = False,
    ) -> PatchApplication:
        """Apply a patch using fuzzy text matching."""
        
        match_result = self._find_best_match(patch.before_code, original_content)
        
        if match_result is None:
            return PatchApplication(
                success=False,
                original_file=original_file,
                patch=patch,
                error="Could not find matching code in original file",
                strategy_used="fuzzy",
            )
        
        match_start, match_end, match_ratio = match_result
        
        if match_ratio < 0.4:
            return PatchApplication(
                success=False,
                original_file=original_file,
                patch=patch,
                error=f"Match confidence too low: {match_ratio:.2f} (need >= 0.4)",
                strategy_used="fuzzy",
            )
        
        matched_content = original_content[match_start:match_end]
        adjusted_after = self._adjust_indentation(patch.after_code, matched_content)
        
        # Ensure proper newlines
        before_part = original_content[:match_start]
        after_part = original_content[match_end:]
        
        if matched_content.endswith('\n') and not adjusted_after.endswith('\n'):
            adjusted_after += '\n'
        if after_part and not after_part.startswith('\n') and not adjusted_after.endswith('\n'):
            adjusted_after += '\n'
        
        patched_content = before_part + adjusted_after + after_part
        
        # Validate
        validation_errors = []
        if validate_python and self.validate_syntax:
            validation_errors = self._validate_python_syntax(patched_content)
            if validation_errors:
                fixed = self._try_fix_python(patched_content, validation_errors)
                if fixed:
                    new_errors = self._validate_python_syntax(fixed)
                    if not new_errors:
                        patched_content = fixed
                        validation_errors = []
        
        if validate_cuda and self.validate_syntax:
            validation_errors = self._validate_cuda_syntax(patched_content)
        
        # Generate diff
        diff = list(difflib.unified_diff(
            original_content.splitlines(keepends=True),
            patched_content.splitlines(keepends=True),
            fromfile=str(original_file),
            tofile=f"{variant_prefix}_v{variant_num}",
            lineterm='',
        ))
        diff_preview = ''.join(diff[:80])
        
        # Create output file
        stem = original_file.stem
        suffix = original_file.suffix
        new_filename = f"{stem}_{variant_prefix}_v{variant_num}{suffix}"
        output_path = output_dir / new_filename
        
        if self.dry_run:
            logger.info(f"[DRY RUN] Would create: {output_path}")
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path.write_text(patched_content)
            if validation_errors:
                logger.warning(f"Created patched file with validation warnings: {output_path}")
            else:
                logger.info(f"Created patched file: {output_path}")
        
        return PatchApplication(
            success=True,
            original_file=original_file,
            patched_file=output_path,
            patch=patch,
            diff_preview=diff_preview,
            validation_errors=validation_errors,
            strategy_used="fuzzy",
        )
    
    def _adjust_indentation(self, new_code: str, original_code: str) -> str:
        """Adjust indentation of new code to match original."""
        original_lines = original_code.split('\n')
        original_indent = 0
        for line in original_lines:
            if line.strip():
                original_indent = len(line) - len(line.lstrip())
                break
        
        new_lines = new_code.split('\n')
        new_indent = 0
        for line in new_lines:
            if line.strip():
                new_indent = len(line) - len(line.lstrip())
                break
        
        indent_diff = original_indent - new_indent
        
        if indent_diff == 0:
            return new_code
        
        adjusted_lines = []
        for line in new_lines:
            if not line.strip():
                adjusted_lines.append(line)
            elif indent_diff > 0:
                adjusted_lines.append(' ' * indent_diff + line)
            else:
                remove = min(-indent_diff, len(line) - len(line.lstrip()))
                adjusted_lines.append(line[remove:])
        
        return '\n'.join(adjusted_lines)
    
    def _find_best_match(
        self, 
        needle: str, 
        haystack: str,
        min_ratio: float = 0.4,
    ) -> Optional[Tuple[int, int, float]]:
        """Find the best fuzzy match for needle in haystack."""
        needle_lines = [l.strip() for l in needle.strip().split('\n') if l.strip()]
        haystack_lines = haystack.split('\n')
        
        if not needle_lines:
            return None
        
        best_match = None
        best_ratio = min_ratio
        
        for window_offset in range(-2, 3):
            window_size = len(needle_lines) + window_offset
            if window_size < 1:
                continue
            
            for i in range(len(haystack_lines) - window_size + 1):
                window = haystack_lines[i:i + window_size]
                window_stripped = [l.strip() for l in window if l.strip()]
                
                if not window_stripped:
                    continue
                
                matcher = difflib.SequenceMatcher(
                    None,
                    '\n'.join(needle_lines),
                    '\n'.join(window_stripped)
                )
                ratio = matcher.ratio()
                
                if ratio > best_ratio:
                    best_ratio = ratio
                    start_pos = sum(len(line) + 1 for line in haystack_lines[:i])
                    end_pos = sum(len(line) + 1 for line in haystack_lines[:i + window_size])
                    end_pos = min(end_pos, len(haystack))
                    best_match = (start_pos, end_pos, ratio)
        
        return best_match
    
    def _validate_python_syntax(self, code: str) -> List[str]:
        """Validate Python syntax."""
        errors = []
        try:
            ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Line {e.lineno}: {e.msg}")
        except Exception as e:
            errors.append(str(e))
        return errors
    
    def _validate_cuda_syntax(self, code: str) -> List[str]:
        """Validate CUDA/C++ syntax (basic checks)."""
        errors = []
        
        open_braces = code.count('{')
        close_braces = code.count('}')
        if open_braces != close_braces:
            errors.append(f"Unbalanced braces: {open_braces} open, {close_braces} close")
        
        open_parens = code.count('(')
        close_parens = code.count(')')
        if open_parens != close_parens:
            errors.append(f"Unbalanced parentheses: {open_parens} open, {close_parens} close")
        
        if '<<<' in code and '>>>' not in code:
            errors.append("CUDA kernel launch syntax incomplete")
        
        return errors
    
    def _try_fix_python(self, code: str, errors: List[str]) -> Optional[str]:
        """Try to fix common Python syntax issues."""
        fixed = code
        fixes_applied = []
        
        patterns = [
            (r'(if\s+[^:]+)\s*\n', r'\1:\n'),
            (r'(for\s+[^:]+)\s*\n', r'\1:\n'),
            (r'(while\s+[^:]+)\s*\n', r'\1:\n'),
            (r'(def\s+[^:]+)\s*\n', r'\1:\n'),
            (r'(class\s+[^:]+)\s*\n', r'\1:\n'),
        ]
        
        for pattern, replacement in patterns:
            new_fixed = re.sub(pattern, replacement, fixed)
            if new_fixed != fixed:
                fixes_applied.append("Added missing colon")
                fixed = new_fixed
        
        fixed = '\n'.join(line.rstrip() for line in fixed.split('\n'))
        
        if fixes_applied:
            logger.info(f"Applied auto-fixes: {', '.join(fixes_applied)}")
        
        return fixed if fixed != code else None


def extract_and_apply_patches(
    llm_analysis_path: Path,
    original_source_path: Path,
    output_dir: Path,
    strategy: str = "ast",
    dry_run: bool = False,
    validate_syntax: bool = True,
) -> List[PatchApplication]:
    """
    Convenience function to extract and apply patches from an LLM analysis file.
    
    Args:
        llm_analysis_path: Path to the LLM analysis markdown file
        original_source_path: Path to the original source file
        output_dir: Directory to write patched files
        strategy: "ast" or "fuzzy"
        dry_run: If True, don't actually write files
        validate_syntax: If True, validate syntax before saving
        
    Returns:
        List of PatchApplication results
    """
    if not llm_analysis_path.exists():
        logger.error(f"LLM analysis file not found: {llm_analysis_path}")
        return []
    
    llm_response = llm_analysis_path.read_text()
    
    applier = LLMPatchApplier(
        strategy=strategy,
        dry_run=dry_run, 
        validate_syntax=validate_syntax
    )
    patches = applier.extract_patches(llm_response)
    
    if not patches:
        logger.info("No patches found in LLM analysis")
        return []
    
    return applier.apply_patches(patches, original_source_path, output_dir)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Apply LLM-suggested patches")
    parser.add_argument("llm_analysis", type=Path, help="Path to LLM analysis markdown")
    parser.add_argument("source_file", type=Path, help="Original source file to patch")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Output directory")
    parser.add_argument("--strategy", choices=["ast", "fuzzy"], default="ast", help="Patching strategy")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--no-validate", action="store_true", help="Skip syntax validation")
    
    args = parser.parse_args()
    
    results = extract_and_apply_patches(
        args.llm_analysis,
        args.source_file,
        args.output_dir,
        strategy=args.strategy,
        dry_run=args.dry_run,
        validate_syntax=not args.no_validate,
    )
    
    print(f"\nPatch Application Results ({args.strategy} strategy):")
    print("=" * 60)
    
    success_count = sum(1 for r in results if r.success)
    print(f"Applied: {success_count}/{len(results)} patches\n")
    
    for i, result in enumerate(results, 1):
        if result.success:
            status = "✓" if not result.validation_errors else "⚠"
            print(f"{status} Patch {i} [{result.strategy_used}]: {result.patched_file}")
            if result.validation_errors:
                print(f"  Warnings: {', '.join(result.validation_errors)}")
            if result.diff_preview:
                print(f"  Diff preview (first 20 lines):")
                for line in result.diff_preview.split('\n')[:20]:
                    print(f"    {line}")
        else:
            print(f"✗ Patch {i} [{result.strategy_used}]: {result.error}")
