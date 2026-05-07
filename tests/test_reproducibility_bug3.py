"""
Bug 3 exploration test: README evaluation command outdated.

CRITICAL: This test MUST FAIL on unfixed code.
Expected failure: README uses --generate-attention-heatmaps which doesn't exist.
"""

import re
import unittest


class TestBug3READMECommand(unittest.TestCase):
    """Property 1: Bug Condition - Outdated README Command."""

    def test_readme_uses_correct_evaluate_flags(self):
        """Test that README evaluation command uses correct CLI flags."""
        # Read README.md
        with open("README.md", "r", encoding="utf-8") as f:
            readme_content = f.read()

        # Bug condition: README contains --generate-attention-heatmaps
        # Expected behavior (after fix): Should NOT contain this flag
        self.assertNotIn(
            "--generate-attention-heatmaps",
            readme_content,
            "README should not use --generate-attention-heatmaps (flag doesn't exist)",
        )

        # Expected behavior: Should use correct flags
        # Find evaluate_camelyon.py command section
        eval_section_match = re.search(
            r"python experiments/evaluate_camelyon\.py.*?```",
            readme_content,
            re.DOTALL,
        )

        if eval_section_match:
            eval_command = eval_section_match.group(0)

            # Should use --heatmaps-dir instead
            self.assertIn(
                "--heatmaps-dir",
                eval_command,
                "README should use --heatmaps-dir flag",
            )


if __name__ == "__main__":
    unittest.main()
