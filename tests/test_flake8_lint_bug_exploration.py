"""
Bug Condition Exploration Test for Flake8 Lint Cleanup

This test MUST FAIL on unfixed code - failure confirms the bug exists.
The test validates Property 1: Bug Condition - Flake8 Lint Violations Across 11 Categories

**Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10, 1.11**

This test runs flake8 on the codebase and verifies that lint violations exist
across 11 distinct categories on UNFIXED code.
"""

import subprocess
from collections import defaultdict


class TestFlake8LintBugExploration:
    """
    Bug condition exploration test for flake8 lint violations.

    This test MUST FAIL on unfixed code to confirm the bug exists.
    Expected behavior: Test FAILS because flake8 reports 60+ violations across 11 categories.
    """

    def run_flake8(self):
        """
        Run flake8 on the codebase with project configuration.

        Returns:
            tuple: (violations_by_code, total_count, stdout)
        """
        # Run flake8 with project configuration using python -m
        result = subprocess.run(
            [
                "python",
                "-m",
                "flake8",
                "src/",
                "tests/",
                "experiments/",
                "--max-line-length=100",
                "--max-complexity=15",
                "--statistics",
            ],
            capture_output=True,
            text=True,
        )

        # Parse violations by error code
        violations_by_code = defaultdict(list)
        total_count = 0

        # Parse output line by line
        for line in result.stdout.splitlines():
            if ":" in line and any(
                code in line
                for code in [
                    "F841",
                    "F541",
                    "E231",
                    "E225",
                    "E712",
                    "E713",
                    "E722",
                    "E731",
                    "F811",
                    "F821",
                    "F402",
                ]
            ):
                # Extract error code from line (format: file:line:col: CODE message)
                parts = line.split(":")
                if len(parts) >= 4:
                    error_part = parts[3].strip()
                    error_code = error_part.split()[0]
                    violations_by_code[error_code].append(line)
                    total_count += 1

        return violations_by_code, total_count, result.stdout

    def test_flake8_violations_exist_property(self):
        """
        Property 1: Bug Condition - Flake8 Lint Violations Across 11 Categories

        **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 1.10, 1.11**

        Test that flake8 reports violations across 11 distinct categories:
        - F841: Unused variables
        - F541: F-strings without placeholders
        - E231: Missing whitespace after comma
        - E225: Missing whitespace around operator
        - E712: Boolean comparison using == or !=
        - E713: Membership test using 'not X in Y'
        - E722: Bare except clause
        - E731: Lambda assignment
        - F811: Redefinition of unused name
        - F821: Undefined name
        - F402: Import shadowed by loop variable

        This test MUST FAIL on unfixed code - failure confirms the bug exists.
        """
        violations_by_code, total_count, stdout = self.run_flake8()

        # Expected violation categories
        expected_categories = [
            "F841",
            "F541",
            "E231",
            "E225",
            "E712",
            "E713",
            "E722",
            "E731",
            "F811",
            "F821",
            "F402",
        ]

        # Document counterexamples found
        print("\n=== FLAKE8 VIOLATIONS FOUND (COUNTEREXAMPLES) ===")
        print(f"Total violations: {total_count}")
        print("\nViolations by category:")

        for code in expected_categories:
            count = len(violations_by_code.get(code, []))
            print(f"\n{code}: {count} violations")
            if count > 0:
                # Show first 5 examples for each category
                for violation in violations_by_code[code][:5]:
                    print(f"  {violation}")
                if count > 5:
                    print(f"  ... and {count - 5} more")

        print("\n=== FULL FLAKE8 OUTPUT ===")
        print(stdout)

        # Bug condition: The system should have ZERO violations
        # On unfixed code, this assertion will FAIL, confirming the bug exists
        assert (
            total_count == 0
        ), f"Bug condition detected: Found {total_count} flake8 violations across {len(violations_by_code)} categories. Expected 0 violations."

        # Additional assertions for each category (all should fail on unfixed code)
        for code in expected_categories:
            count = len(violations_by_code.get(code, []))
            assert (
                count == 0
            ), f"Bug condition detected: Found {count} {code} violations. Expected 0."

    def test_specific_violation_categories(self):
        """
        Test each violation category individually to surface specific counterexamples.

        This generates detailed counterexamples for each violation type.
        """
        violations_by_code, total_count, stdout = self.run_flake8()

        counterexamples = {
            "F841": {
                "description": "Unused variables (scatter, bars, result, metrics)",
                "violations": violations_by_code.get("F841", []),
            },
            "F541": {
                "description": "F-strings without placeholders",
                "violations": violations_by_code.get("F541", []),
            },
            "E231": {
                "description": "Missing whitespace after comma",
                "violations": violations_by_code.get("E231", []),
            },
            "E225": {
                "description": "Missing whitespace around operator",
                "violations": violations_by_code.get("E225", []),
            },
            "E712": {
                "description": "Boolean comparison using == or != (tests/test_camelyon_config.py)",
                "violations": violations_by_code.get("E712", []),
            },
            "E713": {
                "description": "Membership test using 'not X in Y'",
                "violations": violations_by_code.get("E713", []),
            },
            "E722": {
                "description": "Bare except clause (batch_inference.py, train_camelyon.py)",
                "violations": violations_by_code.get("E722", []),
            },
            "E731": {
                "description": "Lambda assignment (test_statistical.py, monitor_training.py)",
                "violations": violations_by_code.get("E731", []),
            },
            "F811": {
                "description": "Redefinition of unused name",
                "violations": violations_by_code.get("F811", []),
            },
            "F821": {
                "description": "Undefined name (run_id in train_pcam.py)",
                "violations": violations_by_code.get("F821", []),
            },
            "F402": {
                "description": "Import shadowed by loop variable",
                "violations": violations_by_code.get("F402", []),
            },
        }

        print("\n=== DETAILED COUNTEREXAMPLES BY CATEGORY ===")
        for code, data in counterexamples.items():
            print(f"\n{code}: {data['description']}")
            print(f"  Count: {len(data['violations'])}")
            if data["violations"]:
                print("  Examples:")
                for violation in data["violations"][:3]:
                    print(f"    {violation}")

        # Document that counterexamples were found
        categories_with_violations = sum(
            1 for data in counterexamples.values() if len(data["violations"]) > 0
        )

        print("\n=== SUMMARY ===")
        print(f"Categories with violations: {categories_with_violations}/11")
        print(f"Total violations: {total_count}")

        # This test documents the bug behavior
        # On unfixed code, we expect violations in multiple categories
        assert (
            categories_with_violations == 0
        ), f"Bug condition detected: Found violations in {categories_with_violations} categories. Expected 0."
