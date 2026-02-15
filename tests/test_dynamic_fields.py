"""
Tests for dynamic field (JSON) support.

Tests introspection of dynamic columns, TOON formatting, validator handling
of dot/bracket notation, and the learning loop for dynamic field patterns.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from mcp_kql_server.utils import SchemaManager
from mcp_kql_server.memory import MemoryManager, TOON_TYPE_MAP
from mcp_kql_server.kql_validator import KQLValidator
from mcp_kql_server.constants import DYNAMIC_TYPE_ACCESSORS, DYNAMIC_ACCESS_PATTERNS
from mcp_kql_server.ai_prompts import build_generation_prompt, FEW_SHOT_EXAMPLES


class TestDynamicFieldIntrospection(unittest.TestCase):
    """Tests for _introspect_dynamic_fields in SchemaManager."""

    def setUp(self):
        self.mm = MagicMock()
        self.sm = SchemaManager(memory_manager=self.mm)

    def test_flat_json_object(self):
        """Introspect a flat JSON object and extract sub-fields."""
        sample_values = [
            '{"displayName": "John Doe", "id": "abc-123", "active": true}'
        ]
        result = self.sm._introspect_dynamic_fields("TestTable", "Properties", sample_values)

        self.assertIn("displayName", result)
        self.assertEqual(result["displayName"]["type"], "string")
        self.assertEqual(result["displayName"]["path"], "Properties.displayName")

        self.assertIn("id", result)
        self.assertEqual(result["id"]["type"], "string")

        self.assertIn("active", result)
        self.assertEqual(result["active"]["type"], "bool")

    def test_nested_json_object(self):
        """Introspect nested objects up to max depth."""
        sample_values = [
            '{"user": {"name": "Alice", "email": "alice@example.com"}, "status": "ok"}'
        ]
        result = self.sm._introspect_dynamic_fields("TestTable", "Data", sample_values)

        self.assertIn("user", result)
        self.assertEqual(result["user"]["type"], "object")
        self.assertIn("nested", result["user"])
        self.assertIn("name", result["user"]["nested"])

        self.assertIn("status", result)
        self.assertEqual(result["status"]["type"], "string")

    def test_array_with_objects(self):
        """Introspect arrays containing objects."""
        sample_values = [
            '[{"name": "Resource1", "type": "VM"}, {"name": "Resource2", "type": "DB"}]'
        ]
        result = self.sm._introspect_dynamic_fields("TestTable", "Resources", sample_values)
        # Should introspect first element's fields
        self.assertIn("name", result)
        self.assertIn("type", result)

    def test_empty_and_null_values(self):
        """Handle empty, null, and invalid JSON gracefully."""
        sample_values = ['', 'null', '{}', 'not-json', None]
        result = self.sm._introspect_dynamic_fields("TestTable", "Col", sample_values)
        self.assertEqual(result, {})

    def test_multiple_samples_merge(self):
        """Merge sub-fields from multiple samples."""
        sample_values = [
            '{"field_a": "val1"}',
            '{"field_a": "val2", "field_b": 42}'
        ]
        result = self.sm._introspect_dynamic_fields("TestTable", "Data", sample_values)
        self.assertIn("field_a", result)
        self.assertIn("field_b", result)
        self.assertEqual(result["field_b"]["type"], "long")

    def test_dict_sample_values(self):
        """Handle sample values that are already deserialized dicts."""
        sample_values = [
            {"deviceId": "abc-def-123", "osVersion": "Windows 11"}
        ]
        result = self.sm._introspect_dynamic_fields("TestTable", "DeviceInfo", sample_values)
        self.assertIn("deviceId", result)
        self.assertIn("osVersion", result)

    def test_guid_detection(self):
        """Detect GUID-type sub-fields."""
        sample_values = [
            '{"resourceId": "12345678-1234-1234-1234-123456789abc"}'
        ]
        result = self.sm._introspect_dynamic_fields("TestTable", "Props", sample_values)
        self.assertEqual(result["resourceId"]["type"], "guid")

    def test_datetime_detection(self):
        """Detect datetime-type sub-fields."""
        sample_values = [
            '{"createdAt": "2024-01-15T10:30:00Z"}'
        ]
        result = self.sm._introspect_dynamic_fields("TestTable", "Props", sample_values)
        self.assertEqual(result["createdAt"]["type"], "datetime")

    def test_max_depth_limit(self):
        """Respect max depth limit for deeply nested objects."""
        deep_json = '{"l1": {"l2": {"l3": {"l4": "deep"}}}}'
        result = self.sm._introspect_dynamic_fields("TestTable", "Col", [deep_json], max_depth=2)
        self.assertIn("l1", result)
        nested = result["l1"].get("nested", {})
        self.assertIn("l2", nested)
        # l3 should not have nested fields since we hit depth 2
        l2_nested = nested.get("l2", {}).get("nested", {})
        if "l3" in l2_nested:
            self.assertNotIn("nested", l2_nested["l3"])


class TestDynamicJoinHints(unittest.TestCase):
    """Tests for _discover_dynamic_join_hints."""

    def setUp(self):
        self.mm = MagicMock()
        self.sm = SchemaManager(memory_manager=self.mm)

    def test_discovers_join_on_matching_column(self):
        """Discover join hint when sub-field name matches another table's column."""
        self.mm._get_database_schema.return_value = [
            {
                "table": "Devices",
                "columns": {
                    "DeviceId": {"data_type": "string"},
                    "DeviceName": {"data_type": "string"}
                }
            }
        ]

        sub_fields = {
            "DeviceId": {"type": "guid", "path": "Properties.DeviceId", "sample": "abc-123"}
        }

        hints = self.sm._discover_dynamic_join_hints("AuditLogs", "Properties", sub_fields, "cluster", "db")
        self.assertEqual(len(hints), 1)
        self.assertEqual(hints[0]["table1"], "AuditLogs")
        self.assertEqual(hints[0]["table2"], "Devices")
        self.assertIn("DeviceId", hints[0]["join_condition"])
        self.assertGreater(hints[0]["confidence"], 0.5)

    def test_no_join_for_non_matching_fields(self):
        """No join hints when sub-fields don't match any column."""
        self.mm._get_database_schema.return_value = [
            {"table": "OtherTable", "columns": {"Unrelated": {"data_type": "string"}}}
        ]

        sub_fields = {"someField": {"type": "string", "path": "Col.someField", "sample": "x"}}
        hints = self.sm._discover_dynamic_join_hints("Table1", "Col", sub_fields, "cluster", "db")
        self.assertEqual(len(hints), 0)


class TestTOONDynamicFormat(unittest.TestCase):
    """Tests for TOON formatting with dynamic sub-fields."""

    def setUp(self):
        self.mm = MemoryManager(":memory:")
        self.mm.semantic_search = MagicMock()
        self.mm.semantic_search.encode.return_value = b'\x00' * (384 * 4)

    def test_toon_includes_dynamic_sub_fields(self):
        """TOON format should show dynamic sub-fields inline."""
        schemas = [{
            "table": "AuditLogs",
            "columns": {
                "TimeGenerated": {"data_type": "datetime"},
                "Properties": {
                    "data_type": "dynamic",
                    "dynamic_fields": {
                        "userId": {"type": "string"},
                        "action": {"type": "string"},
                        "count": {"type": "long"}
                    }
                }
            }
        }]

        toon = self.mm._to_toon(schemas, [], [])
        self.assertIn("TimeGenerated:dt", toon)
        self.assertIn("Properties:dyn{", toon)
        self.assertIn("userId:s", toon)
        self.assertIn("count:l", toon)

    def test_toon_no_sub_fields_regular_dynamic(self):
        """TOON format falls back to plain :dyn when no sub-fields known."""
        schemas = [{
            "table": "T",
            "columns": {
                "Col": {"data_type": "dynamic"}
            }
        }]

        toon = self.mm._to_toon(schemas, [], [])
        self.assertIn("Col:dyn", toon)
        self.assertNotIn("{", toon)

    def test_array_type_in_toon(self):
        """TOON type map includes array type."""
        self.assertIn("array", TOON_TYPE_MAP)
        self.assertEqual(TOON_TYPE_MAP["array"], "arr")


class TestValidatorDynamicFields(unittest.TestCase):
    """Tests for validator handling of dynamic field access patterns."""

    def setUp(self):
        self.mm = MagicMock()
        self.sm = MagicMock()
        self.validator = KQLValidator(self.mm, self.sm)

    def test_extract_dot_notation(self):
        """Extract dynamic field references with dot notation."""
        query = "AuditLogs | where tostring(Properties.userId) == 'admin'"
        refs = self.validator._extract_dynamic_field_references(query)

        base_cols = [r["base_col"] for r in refs]
        self.assertIn("Properties", base_cols)

        paths = [r["field_path"] for r in refs]
        self.assertTrue(any("userId" in p for p in paths))

    def test_extract_bracket_notation(self):
        """Extract dynamic field references with bracket notation."""
        query = 'AuditLogs | project Properties["userId"]'
        refs = self.validator._extract_dynamic_field_references(query)

        base_cols = [r["base_col"] for r in refs]
        self.assertIn("Properties", base_cols)

    def test_extract_nested_dot_notation(self):
        """Extract nested dot notation like Col.Field.SubField."""
        query = "T | project tostring(Data.user.email)"
        refs = self.validator._extract_dynamic_field_references(query)

        self.assertTrue(len(refs) > 0)
        self.assertEqual(refs[0]["base_col"], "Data")
        self.assertIn("user", refs[0]["field_path"])

    def test_skip_kql_keywords(self):
        """Don't extract KQL keywords as dynamic field references."""
        query = "T | join kind=inner OtherTable on Col"
        refs = self.validator._extract_dynamic_field_references(query)

        # "kind.inner" should NOT be extracted
        base_cols = [r["base_col"] for r in refs]
        self.assertNotIn("kind", base_cols)

    def test_skip_dollar_prefix(self):
        """Don't extract $left/$right references."""
        query = "T | join OtherTable on $left.Col == $right.Col"
        refs = self.validator._extract_dynamic_field_references(query)

        base_cols = [r["base_col"] for r in refs]
        for bc in base_cols:
            self.assertFalse(bc.startswith("$"))


class TestDynamicConstants(unittest.TestCase):
    """Tests for dynamic field constants."""

    def test_dynamic_type_accessors(self):
        """DYNAMIC_TYPE_ACCESSORS contains expected mappings."""
        self.assertEqual(DYNAMIC_TYPE_ACCESSORS["string"], "tostring")
        self.assertEqual(DYNAMIC_TYPE_ACCESSORS["long"], "tolong")
        self.assertEqual(DYNAMIC_TYPE_ACCESSORS["datetime"], "todatetime")
        self.assertIsNone(DYNAMIC_TYPE_ACCESSORS["dynamic"])

    def test_dynamic_access_patterns(self):
        """DYNAMIC_ACCESS_PATTERNS contains required regex patterns."""
        self.assertIn("dot_notation", DYNAMIC_ACCESS_PATTERNS)
        self.assertIn("bracket_notation", DYNAMIC_ACCESS_PATTERNS)
        self.assertIn("type_wrapped", DYNAMIC_ACCESS_PATTERNS)


class TestDynamicFewShotExamples(unittest.TestCase):
    """Tests for dynamic field few-shot examples in prompts."""

    def test_has_dynamic_extraction_example(self):
        """FEW_SHOT_EXAMPLES includes a dynamic field extraction example."""
        dynamic_examples = [
            ex for ex in FEW_SHOT_EXAMPLES
            if any(
                col_info.get("data_type") == "dynamic" and col_info.get("dynamic_fields")
                for col_info in ex.get("schema", {}).get("columns", {}).values()
                if isinstance(col_info, dict)
            )
        ]
        self.assertGreaterEqual(len(dynamic_examples), 1)

    def test_dynamic_example_uses_accessor(self):
        """At least one dynamic example uses tostring() or mv-expand."""
        dynamic_queries = [
            ex["query"] for ex in FEW_SHOT_EXAMPLES
            if "tostring(" in ex.get("query", "") or "mv-expand" in ex.get("query", "")
        ]
        self.assertGreaterEqual(len(dynamic_queries), 1)


class TestBuildPromptWithDynamicFields(unittest.TestCase):
    """Tests for build_generation_prompt with dynamic field context."""

    def test_prompt_includes_dynamic_sub_fields(self):
        """build_generation_prompt lists dynamic sub-fields for dynamic columns."""
        schema = {
            "columns": {
                "TimeGenerated": {"data_type": "datetime"},
                "Properties": {
                    "data_type": "dynamic",
                    "dynamic_fields": {
                        "userId": {"type": "string"},
                        "action": {"type": "string"}
                    }
                }
            }
        }

        prompt = build_generation_prompt(
            nl_query="find user actions",
            schema=schema,
            table_name="AuditLogs",
            include_examples=False
        )

        self.assertIn("Sub-fields:", prompt)
        self.assertIn("userId", prompt)
        self.assertIn("action", prompt)
        self.assertIn("tostring(Properties.fieldName)", prompt)

    def test_prompt_without_dynamic_fields(self):
        """build_generation_prompt works normally when no dynamic fields present."""
        schema = {
            "columns": {
                "TimeGenerated": {"data_type": "datetime"},
                "EventId": {"data_type": "int"}
            }
        }

        prompt = build_generation_prompt(
            nl_query="show events",
            schema=schema,
            table_name="Events",
            include_examples=False
        )

        self.assertNotIn("Sub-fields:", prompt)
        self.assertIn("TimeGenerated", prompt)


class TestDynamicFieldLearning(unittest.TestCase):
    """Tests for learning dynamic field patterns from queries."""

    def test_learn_from_dot_notation_query(self):
        """_learn_dynamic_field_patterns extracts dynamic references from queries."""
        from mcp_kql_server.execute_kql import _learn_dynamic_field_patterns

        mock_mm = MagicMock()
        mock_mm._get_database_schema.return_value = [
            {
                "table": "AuditLogs",
                "columns": {
                    "Properties": {
                        "data_type": "dynamic",
                        "dynamic_fields": {}
                    }
                }
            }
        ]

        with patch("mcp_kql_server.execute_kql.get_memory_manager", return_value=mock_mm):
            _learn_dynamic_field_patterns(
                query="AuditLogs | project tostring(Properties.userId)",
                cluster="cluster",
                database="db",
                tables=["AuditLogs"]
            )

        # Verify store_schema was called to update with new sub-field
        mock_mm.store_schema.assert_called_once()
        call_args = mock_mm.store_schema.call_args
        columns = call_args[1].get("schema", call_args[0][3] if len(call_args[0]) > 3 else {}).get("columns", {})
        if columns:
            props = columns.get("Properties", {})
            dyn_fields = props.get("dynamic_fields", {})
            self.assertIn("userId", dyn_fields)


class TestDynamicFieldReliability(unittest.TestCase):
    """
    End-to-end reliability tests for dynamic field discovery.
    Verifies that two tables with dynamic columns containing shared sub-fields
    discover each other's join relationships bidirectionally.
    """

    def setUp(self):
        """Set up a temp-file MemoryManager with two tables, both having dynamic columns."""
        self._tmpfile = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmpfile.close()
        self.mm = MemoryManager(self._tmpfile.name)
        self.mm.semantic_search = MagicMock()
        self.mm.semantic_search.encode.return_value = b'\x00' * (384 * 4)
        self.sm = SchemaManager(memory_manager=self.mm)

        # Store Table A: Events with dynamic Properties containing DeviceId
        self.mm.store_schema("cluster", "db", "Events", {
            "columns": {
                "TimeGenerated": {"data_type": "datetime"},
                "Properties": {
                    "data_type": "dynamic",
                    "sample_values": ['{"DeviceId": "abc-123", "UserId": "user-1"}'],
                    "dynamic_fields": {}
                }
            }
        })

        # Store Table B: Incidents with dynamic Metadata also containing DeviceId
        self.mm.store_schema("cluster", "db", "Incidents", {
            "columns": {
                "IncidentId": {"data_type": "string"},
                "Metadata": {
                    "data_type": "dynamic",
                    "sample_values": ['{"DeviceId": "abc-123", "Severity": "High"}'],
                    "dynamic_fields": {}
                }
            }
        })

    def tearDown(self):
        """Clean up temp file."""
        try:
            os.unlink(self._tmpfile.name)
        except OSError:
            pass

    def test_bidirectional_dynamic_join_discovery(self):
        """
        When Table A has dynamic sub-field DeviceId and Table B also has dynamic
        sub-field DeviceId, enriching Table A should discover a join hint to Table B.
        """
        # First enrich Table B so its dynamic_fields are populated
        columns_b = {
            "IncidentId": {"data_type": "string"},
            "Metadata": {
                "data_type": "dynamic",
                "sample_values": ['{"DeviceId": "abc-123", "Severity": "High"}'],
            }
        }
        self.sm._enrich_dynamic_columns("Incidents", columns_b, "cluster", "db")

        # Store the enriched Table B schema so Table A can see its dynamic_fields
        self.mm.store_schema("cluster", "db", "Incidents", {"columns": columns_b})

        # Clear the in-memory schema cache so Table A sees the updated Table B
        self.mm._schema_cache.clear()

        # Now enrich Table A - it should discover the join via dynamic sub-fields
        columns_a = {
            "TimeGenerated": {"data_type": "datetime"},
            "Properties": {
                "data_type": "dynamic",
                "sample_values": ['{"DeviceId": "abc-123", "UserId": "user-1"}'],
            }
        }
        self.sm._enrich_dynamic_columns("Events", columns_a, "cluster", "db")

        # Verify dynamic_fields were discovered on both tables
        self.assertIn("DeviceId", columns_a["Properties"].get("dynamic_fields", {}))
        self.assertIn("DeviceId", columns_b["Metadata"].get("dynamic_fields", {}))

        # Verify a join hint was stored
        hints = self.mm.get_join_hints(["Events", "Incidents"])
        self.assertTrue(len(hints) > 0, "Expected at least one join hint between Events and Incidents")

        # Verify the hint mentions both tables and DeviceId
        hint_text = " ".join(hints)
        self.assertIn("DeviceId", hint_text)

    def test_join_hints_appear_in_toon_context(self):
        """Join hints discovered from dynamic fields appear in TOON context output."""
        # Manually store a dynamic-discovered join hint
        self.mm.store_join_hint(
            "Events", "Incidents",
            "tostring(Properties.DeviceId) == tostring(Metadata.DeviceId)",
            confidence=0.9
        )

        # Get TOON context
        schemas = self.mm._get_database_schema("cluster", "db")
        join_hints = self.mm.get_join_hints(["Events", "Incidents"])
        toon = self.mm._to_toon(schemas, [], join_hints)

        self.assertIn("Join Hints", toon)
        self.assertIn("DeviceId", toon)

    def test_store_join_hint_respects_confidence(self):
        """store_join_hint should use the provided confidence, not hardcode 1.0."""
        import sqlite3

        self.mm.store_join_hint("TableA", "TableB", "A.id == B.id", confidence=0.7)

        with sqlite3.connect(str(self.mm.db_path)) as conn:
            cursor = conn.execute(
                "SELECT confidence FROM join_hints WHERE table1 = ? AND table2 = ?",
                ("TableA", "TableB")
            )
            row = cursor.fetchone()
            self.assertIsNotNone(row)
            self.assertAlmostEqual(row[0], 0.7, places=1)

    def test_learning_loop_enriches_all_matching_tables(self):
        """
        When a query accesses Properties.DeviceId, the learning loop should
        update ALL tables that have a 'Properties' dynamic column, not just the first.
        """
        from mcp_kql_server.execute_kql import _learn_dynamic_field_patterns

        # Set up two tables that both have a 'Properties' dynamic column
        mock_mm = MagicMock()
        mock_mm._get_database_schema.return_value = [
            {
                "table": "Events",
                "columns": {
                    "Properties": {"data_type": "dynamic", "dynamic_fields": {}}
                }
            },
            {
                "table": "Alerts",
                "columns": {
                    "Properties": {"data_type": "dynamic", "dynamic_fields": {}}
                }
            }
        ]

        with patch("mcp_kql_server.execute_kql.get_memory_manager", return_value=mock_mm):
            _learn_dynamic_field_patterns(
                query="Events | project tostring(Properties.DeviceId)",
                cluster="cluster",
                database="db",
                tables=["Events"]
            )

        # store_schema should be called TWICE - once for each table with Properties column
        self.assertEqual(mock_mm.store_schema.call_count, 2)


if __name__ == "__main__":
    unittest.main()
