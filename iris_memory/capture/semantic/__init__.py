"""
语义提取模块（通道 B）

从 EPISODIC 记忆中通过聚类 + LLM 抽象提取 SEMANTIC 记忆。
"""

from iris_memory.capture.semantic.semantic_clustering import (
    MemoryCluster,
    SemanticClustering,
)
from iris_memory.capture.semantic.semantic_confidence import (
    SemanticConfidenceCalculator,
)
from iris_memory.capture.semantic.semantic_extractor import SemanticExtractor

__all__ = [
    "SemanticClustering",
    "MemoryCluster",
    "SemanticExtractor",
    "SemanticConfidenceCalculator",
]
