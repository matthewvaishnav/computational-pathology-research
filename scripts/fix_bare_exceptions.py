#!/usr/bin/env python3
"""
Fix all bare exception handlers in the codebase.
Replaces 'except:' with specific exception types and proper logging.
"""

import re
from pathlib import Path

# Files to fix with their specific exception types
FIXES = {
    "src/integration/lis/cerner_pathnet_plugin.py": [
        {
            "old": "            except:\n                pass",
            "new": "            except Exception as e:\n                logger.warning(f\"Orders test failed: {type(e).__name__}\")\n                pass",
        }
    ],
    "src/integration/cloud/aws/s3_storage_plugin.py": [
        {
            "old": "            except:\n                pass",
            "new": "            except Exception as e:\n                logger.warning(f\"S3 health check failed: {type(e).__name__}\")\n                pass",
        }
    ],
    "src/integration/cloud/aws/lambda_processing_plugin.py": [
        {
            "old": "            except:\n                pass",
            "new": "            except Exception as e:\n                logger.warning(f\"Lambda health check failed: {type(e).__name__}\")\n                pass",
        }
    ],
    "src/integration/cloud/aws/healthlake_plugin.py": [
        {
            "old": "            except:\n                pass",
            "new": "            except Exception as e:\n                logger.warning(f\"HealthLake health check failed: {type(e).__name__}\")\n                pass",
        }
    ],
    "src/integration/cloud/aws/cloudwatch_monitoring_plugin.py": [
        {
            "old": "            except:\n                pass",
            "new": "            except Exception as e:\n                logger.warning(f\"CloudWatch health check failed: {type(e).__name__}\")\n                pass",
        },
        {
            "old": "                except:\n                    sns_accessible = False",
            "new": "                except Exception as e:\n                    logger.warning(f\"SNS health check failed: {type(e).__name__}\")\n                    sns_accessible = False",
        },
    ],
    "scripts/demo_foundation_model.py": [
        {
            "old": "            except:\n                print(\"Plot display not available in current environment\")",
            "new": "            except (ImportError, RuntimeError) as e:\n                print(\"Plot display not available in current environment\")",
        }
    ],
    "scripts/histocore-admin.py": [
        {
            "old": "        except:\n            results[name] = \"Error\"",
            "new": "        except Exception as e:\n            results[name] = \"Error\"\n            print(f\"Check failed for {name}: {type(e).__name__}\")",
        }
    ],
    "scripts/pilot_deployment_manager.py": [
        {
            "old": "            except:\n                k8s_config.load_kube_config()",
            "new": "            except Exception as e:\n                # Fall back to kubeconfig\n                k8s_config.load_kube_config()",
        }
    ],
    "scripts/download_pmc_pathology.py": [
        {
            "old": "            except:\n                continue",
            "new": "            except Exception as e:\n                print(f\"Failed to download figure: {type(e).__name__}\")\n                continue",
        }
    ],
}


def fix_file(filepath: str, fixes: list):
    """Apply fixes to a file"""
    path = Path(filepath)
    if not path.exists():
        print(f"⚠️  File not found: {filepath}")
        return False

    content = path.read_text(encoding="utf-8")
    original_content = content

    for fix in fixes:
        if fix["old"] in content:
            content = content.replace(fix["old"], fix["new"], 1)
            print(f"✓ Fixed bare except in {filepath}")
        else:
            print(f"⚠️  Pattern not found in {filepath}")

    if content != original_content:
        path.write_text(content, encoding="utf-8")
        return True
    return False


def main():
    """Fix all bare exceptions"""
    print("Fixing bare exception handlers...\n")

    fixed_count = 0
    for filepath, fixes in FIXES.items():
        if fix_file(filepath, fixes):
            fixed_count += 1

    print(f"\n✓ Fixed {fixed_count} files")


if __name__ == "__main__":
    main()
