"""
Bug 3 preservation test: Other README commands work correctly.

Property 2: Preservation - Other README Commands

IMPORTANT: This test follows observation-first methodology.
- Observe behavior on UNFIXED code for non-buggy README commands
- Write property-based tests capturing observed behavior patterns
- EXPECTED OUTCOME: Tests PASS (confirms baseline behavior to preserve)

**Validates: Requirements 3.7, 3.8, 3.9**
"""

import re
import unittest
from typing import List


class TestBug3PreservationOtherREADMECommands(unittest.TestCase):
    """Property 2: Preservation - Other README Commands work as documented."""

    def setUp(self):
        """Read README.md once for all tests."""
        with open("README.md", "r", encoding="utf-8") as f:
            self.readme_content = f.read()

    def _extract_bash_commands(self, content: str) -> List[str]:
        """Extract all bash commands from markdown code blocks."""
        # Find all bash code blocks
        bash_blocks = re.findall(r"```bash\n(.*?)```", content, re.DOTALL)

        commands = []
        for block in bash_blocks:
            # Split by lines and filter out comments and empty lines
            lines = block.strip().split("\n")
            current_command = []

            for line in lines:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith("#"):
                    continue

                # If line ends with backslash, it's a continuation
                if line.endswith("\\"):
                    current_command.append(line[:-1].strip())
                else:
                    # Complete command
                    current_command.append(line)
                    # Join multi-line command
                    full_command = " ".join(current_command).strip()
                    if full_command:
                        commands.append(full_command)
                    current_command = []

            # Handle any remaining command
            if current_command:
                full_command = " ".join(current_command).strip()
                if full_command:
                    commands.append(full_command)

        return commands

    def _is_buggy_command(self, command: str) -> bool:
        """Check if this is the buggy CAMELYON evaluation command."""
        # The buggy command is the one with --generate-attention-heatmaps
        return "evaluate_camelyon.py" in command and "--generate-attention-heatmaps" in command

    def _is_pcam_training_command(self, command: str) -> bool:
        """Check if this is a PCam training command."""
        return "train_pcam.py" in command

    def _is_pcam_evaluation_command(self, command: str) -> bool:
        """Check if this is a PCam evaluation command."""
        return "evaluate_pcam.py" in command

    def _is_pcam_comparison_command(self, command: str) -> bool:
        """Check if this is a PCam baseline comparison command."""
        return "compare_pcam_baselines.py" in command

    def _is_camelyon_training_command(self, command: str) -> bool:
        """Check if this is a CAMELYON training command."""
        return "train_camelyon.py" in command

    def _is_data_download_command(self, command: str) -> bool:
        """Check if this is a data download command."""
        return "download_pcam.py" in command or "generate_synthetic" in command

    def test_pcam_training_commands_use_valid_flags(self):
        """
        Test that PCam training commands use valid CLI flags.

        **Validates: Requirements 3.7** - PCam training commands work as documented
        """
        commands = self._extract_bash_commands(self.readme_content)
        pcam_training_commands = [cmd for cmd in commands if self._is_pcam_training_command(cmd)]

        # Should have at least one PCam training command
        self.assertGreater(
            len(pcam_training_commands),
            0,
            "README should contain PCam training commands",
        )

        # All PCam training commands should use --config flag
        for cmd in pcam_training_commands:
            if "train_pcam.py" in cmd and not cmd.strip().endswith("train_pcam.py"):
                # If there are arguments, should use --config
                self.assertIn(
                    "--config",
                    cmd,
                    f"PCam training command should use --config flag: {cmd}",
                )

    def test_pcam_evaluation_commands_use_valid_flags(self):
        """
        Test that PCam evaluation commands use valid CLI flags.

        **Validates: Requirements 3.8** - Other evaluation scripts work as documented
        """
        commands = self._extract_bash_commands(self.readme_content)
        pcam_eval_commands = [cmd for cmd in commands if self._is_pcam_evaluation_command(cmd)]

        # Should have at least one PCam evaluation command
        self.assertGreater(
            len(pcam_eval_commands),
            0,
            "README should contain PCam evaluation commands",
        )

        # Valid flags for evaluate_pcam.py
        valid_flags = [
            "--checkpoint",
            "--data-root",
            "--output-dir",
            "--compute-bootstrap-ci",
            "--bootstrap-samples",
            "--split",
            "--device",
        ]

        for cmd in pcam_eval_commands:
            # Should use at least --checkpoint flag
            self.assertIn(
                "--checkpoint",
                cmd,
                f"PCam evaluation command should use --checkpoint flag: {cmd}",
            )

            # Extract all flags from command
            flags = re.findall(r"--[\w-]+", cmd)

            # All flags should be valid
            for flag in flags:
                self.assertIn(
                    flag,
                    valid_flags,
                    f"PCam evaluation command uses invalid flag {flag}: {cmd}",
                )

    def test_pcam_comparison_commands_use_valid_flags(self):
        """
        Test that PCam baseline comparison commands use valid CLI flags.

        **Validates: Requirements 3.7** - PCam commands work as documented
        """
        commands = self._extract_bash_commands(self.readme_content)
        comparison_commands = [cmd for cmd in commands if self._is_pcam_comparison_command(cmd)]

        # Should have at least one comparison command
        self.assertGreater(
            len(comparison_commands),
            0,
            "README should contain PCam comparison commands",
        )

        # Valid flags for compare_pcam_baselines.py
        valid_flags = [
            "--configs",
            "--output",
            "--compute-bootstrap-ci",
            "--quick-test",
        ]

        for cmd in comparison_commands:
            # Should use --configs flag
            self.assertIn(
                "--configs",
                cmd,
                f"Comparison command should use --configs flag: {cmd}",
            )

            # Extract all flags from command
            flags = re.findall(r"--[\w-]+", cmd)

            # All flags should be valid
            for flag in flags:
                self.assertIn(
                    flag,
                    valid_flags,
                    f"Comparison command uses invalid flag {flag}: {cmd}",
                )

    def test_camelyon_training_commands_use_valid_flags(self):
        """
        Test that CAMELYON training commands use valid CLI flags.

        **Validates: Requirements 3.7** - Other commands work as documented
        """
        commands = self._extract_bash_commands(self.readme_content)
        camelyon_training_commands = [
            cmd for cmd in commands if self._is_camelyon_training_command(cmd)
        ]

        # Should have at least one CAMELYON training command
        self.assertGreater(
            len(camelyon_training_commands),
            0,
            "README should contain CAMELYON training commands",
        )

        # All CAMELYON training commands should use --config flag
        for cmd in camelyon_training_commands:
            if "train_camelyon.py" in cmd and not cmd.strip().endswith("train_camelyon.py"):
                # If there are arguments, should use --config
                self.assertIn(
                    "--config",
                    cmd,
                    f"CAMELYON training command should use --config flag: {cmd}",
                )

    def test_non_buggy_camelyon_evaluation_commands_use_valid_flags(self):
        """
        Test that non-buggy CAMELYON evaluation commands use valid CLI flags.

        **Validates: Requirements 3.9** - Heatmap generation with current interface works
        """
        commands = self._extract_bash_commands(self.readme_content)

        # Find evaluate_camelyon.py commands that are NOT the buggy one
        camelyon_eval_commands = [
            cmd
            for cmd in commands
            if "evaluate_camelyon.py" in cmd and not self._is_buggy_command(cmd)
        ]

        # Valid flags for evaluate_camelyon.py (current interface)
        valid_flags = [
            "--checkpoint",
            "--data-root",
            "--split",
            "--output-dir",
            "--device",
            "--tile-scores-dir",
            "--heatmaps-dir",
            "--heatmap-downsample",
            "--save-predictions-csv",
        ]

        for cmd in camelyon_eval_commands:
            # Should use --checkpoint flag
            self.assertIn(
                "--checkpoint",
                cmd,
                f"CAMELYON evaluation command should use --checkpoint flag: {cmd}",
            )

            # Should NOT use the outdated flag
            self.assertNotIn(
                "--generate-attention-heatmaps",
                cmd,
                f"Non-buggy command should not use --generate-attention-heatmaps: {cmd}",
            )

            # Extract all flags from command
            flags = re.findall(r"--[\w-]+", cmd)

            # All flags should be valid
            for flag in flags:
                self.assertIn(
                    flag,
                    valid_flags,
                    f"CAMELYON evaluation command uses invalid flag {flag}: {cmd}",
                )

    def test_data_download_commands_use_valid_flags(self):
        """
        Test that data download commands use valid CLI flags.

        **Validates: Requirements 3.7** - Other commands work as documented
        """
        commands = self._extract_bash_commands(self.readme_content)
        download_commands = [cmd for cmd in commands if self._is_data_download_command(cmd)]

        # Should have at least one download command
        self.assertGreater(
            len(download_commands),
            0,
            "README should contain data download commands",
        )

        # Valid flags for download scripts
        # Updated to include all flags currently supported by download scripts:
        # - download_pcam.py: --output-dir, --keep-compressed, --skip-existing
        # - download_pcam_manual.py: --root_dir
        # - generate_synthetic_test_data.py: --dataset, --samples
        # - generate_synthetic_pcam.py: --root_dir, --train_size, --val_size, --test_size, --image_size
        # - generate_synthetic_camelyon.py: --output-dir, --num-train, --num-val, --num-test, --num-patches, --feature-dim, --seed
        valid_flags = [
            "--output-dir",
            "--data-root",
            "--root_dir",
            "--keep-compressed",
            "--skip-existing",
            "--dataset",
            "--samples",
            "--train_size",
            "--val_size",
            "--test_size",
            "--image_size",
            "--num-train",
            "--num-val",
            "--num-test",
            "--num-patches",
            "--feature-dim",
            "--seed",
        ]

        for cmd in download_commands:
            # Extract all flags from command
            flags = re.findall(r"--[\w-]+", cmd)

            # All flags should be valid
            for flag in flags:
                self.assertIn(
                    flag,
                    valid_flags,
                    f"Download command uses invalid flag {flag}: {cmd}",
                )

    def test_readme_contains_no_other_outdated_flags(self):
        """
        Test that README doesn't contain other known outdated flags.

        **Validates: Requirements 3.8** - Other evaluation scripts work as documented
        """
        # Known outdated flags that should not appear
        outdated_flags = [
            "--generate-attention-heatmaps",  # The main bug
            "--attention-weights",  # Hypothetical outdated flag
            "--export-attention",  # Hypothetical outdated flag
        ]

        commands = self._extract_bash_commands(self.readme_content)

        # Exclude the known buggy command
        non_buggy_commands = [cmd for cmd in commands if not self._is_buggy_command(cmd)]

        for cmd in non_buggy_commands:
            for outdated_flag in outdated_flags:
                self.assertNotIn(
                    outdated_flag,
                    cmd,
                    f"Non-buggy command should not use outdated flag {outdated_flag}: {cmd}",
                )

    def test_all_python_script_commands_are_valid_paths(self):
        """
        Test that all Python script paths in commands are valid.

        **Validates: Requirements 3.7, 3.8** - Commands work as documented
        """
        commands = self._extract_bash_commands(self.readme_content)

        # Extract Python script paths
        python_scripts = []
        for cmd in commands:
            # Match patterns like "python experiments/train_pcam.py"
            match = re.search(r"python\s+([\w/]+\.py)", cmd)
            if match:
                python_scripts.append(match.group(1))

        # Should have found some Python scripts
        self.assertGreater(
            len(python_scripts),
            0,
            "README should contain Python script commands",
        )

        # All scripts should be in expected directories
        valid_prefixes = [
            "experiments/",
            "scripts/",
            "examples/",
        ]

        for script in python_scripts:
            has_valid_prefix = any(script.startswith(prefix) for prefix in valid_prefixes)
            self.assertTrue(
                has_valid_prefix,
                f"Python script should be in experiments/, scripts/, or examples/: {script}",
            )


if __name__ == "__main__":
    unittest.main()
