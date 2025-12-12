"""
Book Citation System - Provides citations from AI Systems Performance Engineering book.

This module searches through book/ch*.md files to find relevant content
and provide citations for optimization techniques and concepts.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass


# Book chapter topics for targeted search (aligned with book/chXX.md titles)
CHAPTER_TOPICS = {
    "ch01": ["introduction", "overview", "performance engineering", "AI systems"],
    "ch02": ["hardware overview", "GPU", "CPU", "interconnects", "Blackwell", "Grace", "NVLink", "HBM"],
    "ch03": ["OS tuning", "docker", "kubernetes", "containers", "system configuration", "NUMA", "cgroups"],
    "ch04": ["distributed training", "distributed inference", "multi-GPU", "NCCL", "NVLink", "NIXL", "overlap"],
    "ch05": ["storage", "I/O", "GPUDirect Storage", "NVMe", "data loading", "input pipeline"],
    "ch06": ["GPU architecture", "CUDA", "SIMT", "warps", "SMs", "memory hierarchy", "tensor cores"],
    "ch07": ["memory access", "coalescing", "shared memory", "bank conflicts", "caching", "HBM"],
    "ch08": ["occupancy", "warp efficiency", "ILP", "warp divergence", "software pipelining"],
    "ch09": ["arithmetic intensity", "roofline", "tiling", "kernel efficiency", "vectorization"],
    "ch10": ["intra-kernel pipelining", "cuda::pipeline", "TMA", "thread-block clusters", "warp specialization"],
    "ch11": ["CUDA streams", "synchronization", "inter-kernel overlap", "concurrency", "PDL"],
    "ch12": ["dynamic scheduling", "CUDA graphs", "irregular workloads", "persistent kernels"],
    "ch13": ["PyTorch profiling", "torch.profiler", "nsight", "DDP", "FSDP", "scaling"],
    "ch14": ["torch.compile", "TorchDynamo", "Inductor", "Triton", "kernel fusion", "autotuning"],
    "ch15": ["multinode inference", "parallelism", "tensor parallel", "pipeline parallel", "KV cache", "NIXL"],
    "ch16": ["production inference", "profiling", "debugging", "vLLM", "PagedAttention", "continuous batching"],
    "ch17": ["disaggregated prefill", "decode", "routing", "continuous batching", "speculative decoding"],
    "ch18": ["attention optimization", "FlashAttention", "FlashMLA", "KV transfer", "precision"],
    "ch19": ["dynamic inference", "adaptive kernels", "quantization policies", "runtime autotuning", "RL control"],
    "ch20": ["AI-assisted optimization", "autotuning", "reinforcement learning", "AlphaTensor", "automation"],
}

# Technique to chapter mapping (aligned with book/chXX.md)
TECHNIQUE_CHAPTERS = {
    # Memory optimizations
    "coalescing": ["ch07"],
    "shared_memory": ["ch07", "ch10"],
    "bank_conflicts": ["ch07"],
    "memory_bandwidth": ["ch06", "ch07"],
    "hbm": ["ch06", "ch07"],
    
    # Compute optimizations
    "occupancy": ["ch08"],
    "warp_divergence": ["ch08"],
    "ilp": ["ch08", "ch09"],
    "loop_unrolling": ["ch08", "ch09"],
    
    # Tensor cores / GEMM
    "tensor_cores": ["ch06", "ch09", "ch10"],
    "wmma": ["ch06", "ch09"],
    "mixed_precision": ["ch06", "ch09", "ch18"],
    "gemm": ["ch06", "ch09", "ch10"],
    
    # Attention / KV
    "flash_attention": ["ch14", "ch18", "ch19"],
    "flash attention": ["ch14", "ch18", "ch19"],
    "attention": ["ch18", "ch19"],
    "sdpa": ["ch14", "ch18", "ch19"],
    "kv_cache": ["ch15", "ch17", "ch18"],
    
    # Kernel optimization
    "kernel_fusion": ["ch14"],
    "triton": ["ch14"],
    "custom_kernels": ["ch14"],
    
    # Compilation
    "torch_compile": ["ch13", "ch14"],
    "inductor": ["ch14"],
    "cuda_graphs": ["ch12", "ch13"],
    
    # Quantization / low precision
    "fp8": ["ch18", "ch19"],
    "int8": ["ch18", "ch19"],
    "int4": ["ch18", "ch19"],
    "quantization": ["ch18", "ch19"],
    "awq": ["ch18", "ch19"],
    "gptq": ["ch18", "ch19"],
    
    # Distributed training / parallelism
    "ddp": ["ch04", "ch13"],
    "data_parallel": ["ch04", "ch13"],
    "tensor_parallel": ["ch04", "ch15"],
    "pipeline_parallel": ["ch04", "ch15"],
    "fsdp": ["ch04", "ch13"],
    "zero": ["ch13"],
    
    # Communication
    "nccl": ["ch04", "ch13", "ch15"],
    "all_reduce": ["ch04", "ch13"],
    "collective": ["ch04", "ch13"],
    "nvlink": ["ch02", "ch04", "ch15"],
    
    # Inference
    "vllm": ["ch15", "ch16", "ch17"],
    "continuous_batching": ["ch15", "ch16", "ch17"],
    "paged_attention": ["ch16", "ch17", "ch18"],
    "speculative_decoding": ["ch15", "ch16", "ch17"],
    
    # RL / AI-assisted optimization
    "rlhf": ["ch20"],
    "ppo": ["ch19", "ch20"],
    "dpo": ["ch20"],
    "grpo": ["ch20"],
}


@dataclass
class BookCitation:
    """A citation from the book."""
    chapter: str
    chapter_title: str
    section: str
    content: str
    relevance_score: float
    line_number: int


class BookIndex:
    """Index and search the book chapters."""
    
    def __init__(self, book_dir: Optional[Path] = None):
        if book_dir is None:
            # Find book directory relative to this file
            self.book_dir = Path(__file__).resolve().parent.parent / "book"
        else:
            self.book_dir = Path(book_dir)
        
        self._chapter_cache: Dict[str, str] = {}
        self._chapter_titles: Dict[str, str] = {}
        self._load_chapter_titles()
    
    def _load_chapter_titles(self):
        """Load chapter titles from the first line of each chapter."""
        if not self.book_dir.exists():
            return
            
        for chapter_file in sorted(self.book_dir.glob("ch*.md")):
            chapter_id = chapter_file.stem
            try:
                with open(chapter_file, 'r') as f:
                    first_line = f.readline().strip()
                    # Extract title from "# Chapter X: Title" format
                    if first_line.startswith('#'):
                        title = first_line.lstrip('#').strip()
                        self._chapter_titles[chapter_id] = title
            except Exception:
                self._chapter_titles[chapter_id] = f"Chapter {chapter_id[2:]}"
    
    def _load_chapter(self, chapter_id: str) -> str:
        """Load a chapter's content, with caching."""
        if chapter_id in self._chapter_cache:
            return self._chapter_cache[chapter_id]
        
        chapter_file = self.book_dir / f"{chapter_id}.md"
        if not chapter_file.exists():
            return ""
        
        try:
            content = chapter_file.read_text()
            self._chapter_cache[chapter_id] = content
            return content
        except Exception:
            return ""
    
    def search(self, query: str, max_results: int = 3) -> List[BookCitation]:
        """Search the book for relevant content.
        
        Args:
            query: The search query (technique name, concept, or question)
            max_results: Maximum number of citations to return
            
        Returns:
            List of BookCitation objects sorted by relevance
        """
        results = []
        
        # Normalize query
        query_lower = query.lower()
        query_terms = set(re.findall(r'\w+', query_lower))
        
        # Determine which chapters to search
        chapters_to_search = self._get_relevant_chapters(query_lower)
        
        for chapter_id in chapters_to_search:
            content = self._load_chapter(chapter_id)
            if not content:
                continue
            
            # Search for relevant sections
            citations = self._search_chapter(chapter_id, content, query_terms)
            results.extend(citations)
        
        # Sort by relevance and return top results
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:max_results]
    
    def _get_relevant_chapters(self, query: str) -> List[str]:
        """Determine which chapters are most relevant to the query."""
        query_lower = query.lower()
        # Check technique mapping first
        for technique, chapters in TECHNIQUE_CHAPTERS.items():
            if technique.replace('_', ' ') in query_lower or technique in query_lower:
                return chapters
        
        # Otherwise search all chapters by topic
        relevant = []
        for chapter_id, topics in CHAPTER_TOPICS.items():
            for topic in topics:
                if topic.lower() in query_lower:
                    relevant.append(chapter_id)
                    break
        
        # If no specific match, search all chapters
        if not relevant:
            relevant = [f"ch{i:02d}" for i in range(1, 21)]
        
        return relevant
    
    def _search_chapter(self, chapter_id: str, content: str, query_terms: set) -> List[BookCitation]:
        """Search a single chapter for relevant content."""
        citations = []
        
        # Split into sections (by ## headers)
        sections = re.split(r'\n(?=##\s)', content)
        
        for section in sections:
            lines = section.split('\n')
            if not lines:
                continue
            
            # Get section title
            section_title = lines[0].lstrip('#').strip() if lines[0].startswith('#') else "Introduction"
            section_text = '\n'.join(lines)
            
            # Calculate relevance score
            score = self._calculate_relevance(section_text.lower(), query_terms)
            
            if score > 0.1:  # Threshold for relevance
                # Extract most relevant paragraph
                relevant_content = self._extract_relevant_content(section_text, query_terms)
                
                if relevant_content:
                    citations.append(BookCitation(
                        chapter=chapter_id,
                        chapter_title=self._chapter_titles.get(chapter_id, chapter_id),
                        section=section_title,
                        content=relevant_content,
                        relevance_score=score,
                        line_number=self._find_line_number(content, relevant_content),
                    ))
        
        return citations
    
    def _calculate_relevance(self, text: str, query_terms: set) -> float:
        """Calculate relevance score based on term frequency."""
        if not query_terms:
            return 0.0
        
        text_terms = set(re.findall(r'\w+', text))
        matches = query_terms & text_terms
        
        # Base score from term overlap
        score = len(matches) / len(query_terms)
        
        # Boost for exact phrase matches
        for term in query_terms:
            if term in text:
                score += 0.1
        
        return min(score, 1.0)
    
    def _extract_relevant_content(self, section_text: str, query_terms: set, max_length: int = 500) -> str:
        """Extract the most relevant paragraph from a section."""
        paragraphs = section_text.split('\n\n')
        
        best_para = ""
        best_score = 0
        
        for para in paragraphs:
            para = para.strip()
            if len(para) < 50:  # Skip very short paragraphs
                continue
            
            score = self._calculate_relevance(para.lower(), query_terms)
            if score > best_score:
                best_score = score
                best_para = para
        
        # Truncate if too long
        if len(best_para) > max_length:
            best_para = best_para[:max_length] + "..."
        
        return best_para
    
    def _find_line_number(self, content: str, snippet: str) -> int:
        """Find the line number of a snippet in the content."""
        lines = content.split('\n')
        snippet_start = snippet[:50] if len(snippet) > 50 else snippet
        
        for i, line in enumerate(lines, 1):
            if snippet_start in line:
                return i
        
        return 0
    
    def get_chapter_overview(self, chapter_id: str) -> Dict[str, Any]:
        """Get an overview of a chapter's content."""
        content = self._load_chapter(chapter_id)
        if not content:
            return {"error": f"Chapter {chapter_id} not found"}
        
        # Extract sections
        sections = []
        for match in re.finditer(r'^##\s+(.+)$', content, re.MULTILINE):
            sections.append(match.group(1))
        
        # Count code blocks
        code_blocks = len(re.findall(r'```\w*\n', content))
        
        return {
            "chapter": chapter_id,
            "title": self._chapter_titles.get(chapter_id, chapter_id),
            "sections": sections,
            "code_blocks": code_blocks,
            "word_count": len(content.split()),
        }


def get_citations(query: str, max_results: int = 3) -> List[BookCitation]:
    """Convenience function to search the book.
    
    Args:
        query: The search query
        max_results: Maximum number of results
        
    Returns:
        List of BookCitation objects
    """
    index = BookIndex()
    return index.search(query, max_results)


def format_citation(citation: BookCitation) -> str:
    """Format a citation for display."""
    return f"""
📖 **{citation.chapter_title}** ({citation.chapter}.md)
   Section: {citation.section}
   
   {citation.content}
   
   [Line {citation.line_number}]
"""


def format_citations(citations: List[BookCitation]) -> str:
    """Format multiple citations for display."""
    if not citations:
        return "No relevant book citations found."
    
    output = "\n📚 **Book References:**\n"
    output += "=" * 60 + "\n"
    
    for i, citation in enumerate(citations, 1):
        output += f"\n{i}. **{citation.chapter_title}** ({citation.chapter}.md)\n"
        output += f"   Section: {citation.section}\n"
        output += f"   \n   {citation.content}\n"
        output += f"   \n   [Line {citation.line_number}, Relevance: {citation.relevance_score:.0%}]\n"
        output += "-" * 60 + "\n"
    
    return output
