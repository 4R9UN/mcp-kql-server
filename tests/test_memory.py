"""
Unit tests for the unified schema memory module.

Author: Arjun Trivedi
Email: arjuntrivedi42@yahoo.com
"""

import unittest

from mcp_kql_server.memory import (
    get_memory_manager,
)


class TestMemoryManager(unittest.TestCase):
    """Test cases for Memory Manager functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.memory_manager = get_memory_manager()

    def test_memory_manager_initialization(self):
        """Test that memory manager initializes correctly."""
        self.assertIsNotNone(self.memory_manager)
        self.assertHasAttr(self.memory_manager, "corpus")

    def test_get_corpus_access(self):
        """Test accessing corpus through memory manager."""
        corpus = self.memory_manager.corpus
        self.assertIsInstance(corpus, dict)

    def test_get_memory_stats(self):
        """Test getting memory statistics."""
        stats = self.memory_manager.get_memory_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn("schema_count", stats)
        self.assertIn("query_count", stats)
        self.assertIn("db_size_bytes", stats)

    def assertHasAttr(self, obj, attr):
        """Helper method to check if object has attribute."""
        self.assertTrue(hasattr(obj, attr), f"Object does not have attribute '{attr}'")


if __name__ == "__main__":
    unittest.main()
