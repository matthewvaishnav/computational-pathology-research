"""
Secure subprocess execution utilities.

Provides safe wrappers for subprocess calls to prevent command injection.
"""

import logging
import subprocess
from typing import List, Optional, Union

logger = logging.getLogger(__name__)


def run_command_safe(
    cmd: List[str],
    timeout: Optional[int] = None,
    check: bool = True,
    capture_output: bool = True,
) -> subprocess.CompletedProcess:
    """Execute command safely without shell=True.

    Args:
        cmd: Command as list of strings (NOT a single string)
        timeout: Timeout in seconds
        check: Raise exception on non-zero exit
        capture_output: Capture stdout/stderr

    Returns:
        CompletedProcess result

    Raises:
        ValueError: If cmd is not a list
        subprocess.CalledProcessError: If command fails and check=True
        subprocess.TimeoutExpired: If timeout exceeded
    """
    if not isinstance(cmd, list):
        raise ValueError("Command must be a list, not a string. This prevents shell injection.")

    # Validate no shell metacharacters in arguments
    dangerous_chars = [";", "|", "&", "$", "`", "(", ")", "<", ">", "\n"]
    for arg in cmd:
        if any(char in str(arg) for char in dangerous_chars):
            logger.warning(f"Potentially dangerous characters in command argument: {arg}")

    logger.info(f"Executing command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            shell=False,  # NEVER use shell=True
            capture_output=capture_output,
            text=True,
            timeout=timeout,
            check=check,
        )
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}: {' '.join(cmd)}")
        logger.error(f"stderr: {e.stderr}")
        raise
    except subprocess.TimeoutExpired as e:
        logger.error(f"Command timed out after {timeout}s: {' '.join(cmd)}")
        raise


def validate_command_path(cmd_path: str, allowed_commands: Optional[List[str]] = None) -> bool:
    """Validate command is in allowed list.

    Args:
        cmd_path: Path or name of command
        allowed_commands: List of allowed command names

    Returns:
        True if command is allowed

    Raises:
        ValueError: If command not in allowed list
    """
    import os

    cmd_name = os.path.basename(cmd_path)

    if allowed_commands and cmd_name not in allowed_commands:
        raise ValueError(f"Command '{cmd_name}' not in allowed list: {allowed_commands}")

    return True
