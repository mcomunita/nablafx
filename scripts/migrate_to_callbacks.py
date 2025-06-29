#!/usr/bin/env python3
"""
Migration script to add use_callbacks: true to all system configurations in cfg-new.

This script updates all BlackBoxSystem and GreyBoxSystem configurations to enable
the new callback-based logging system.
"""

import os
import re
import yaml
from pathlib import Path


def update_system_config(file_path: Path) -> bool:
    """
    Update a system configuration file to add use_callbacks: true.

    Returns True if the file was modified, False otherwise.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check if this is a system config file
        if "BlackBoxSystem" not in content and "GreyBoxSystem" not in content:
            return False

        # Check if use_callbacks is already present
        if "use_callbacks:" in content:
            print(f"⏭️  Skipping {file_path.name} - use_callbacks already present")
            return False

        # Find the pattern: class_path: nablafx.system.XxxSystem
        # followed by init_args: and then lr:
        pattern = r"(class_path: nablafx\.system\.(BlackBoxSystem|GreyBoxSystem)\s+init_args:\s+)(lr: [^\n]+)"

        def replacement(match):
            return f"{match.group(1)}{match.group(3)}\n    use_callbacks: true"

        updated_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

        # If no change was made, try a different pattern
        if updated_content == content:
            # Try pattern with log_media_every_n_steps
            pattern2 = r"(class_path: nablafx\.system\.(BlackBoxSystem|GreyBoxSystem)\s+init_args:\s+lr: [^\n]+\s+)(log_media_every_n_steps: [^\n]+)"

            def replacement2(match):
                return f"{match.group(1)}use_callbacks: true\n    # {match.group(3)} # Now handled by AudioLoggingCallback"

            updated_content = re.sub(pattern2, replacement2, content, flags=re.MULTILINE)

        # If still no change, add use_callbacks after lr line
        if updated_content == content:
            pattern3 = r"(class_path: nablafx\.system\.(BlackBoxSystem|GreyBoxSystem)\s+init_args:\s+lr: [^\n]+)"

            def replacement3(match):
                return f"{match.group(1)}\n    use_callbacks: true"

            updated_content = re.sub(pattern3, replacement3, content, flags=re.MULTILINE)

        if updated_content != content:
            # Create backup
            backup_path = file_path.with_suffix(".yaml.backup")
            with open(backup_path, "w", encoding="utf-8") as f:
                f.write(content)

            # Write updated content
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(updated_content)

            print(f"✅ Updated {file_path.name}")
            return True
        else:
            print(f"⚠️  No changes made to {file_path.name}")
            return False

    except Exception as e:
        print(f"❌ Error processing {file_path.name}: {e}")
        return False


def main():
    """Main migration function."""
    cfg_new_path = Path("cfg-new")

    if not cfg_new_path.exists():
        print("❌ cfg-new directory not found!")
        return

    print("🔄 Migrating system configurations to use callbacks...")
    print("=" * 60)

    # Find all YAML files in cfg-new
    yaml_files = list(cfg_new_path.rglob("*.yaml"))
    system_files = []
    updated_count = 0

    # Filter for system configuration files
    for yaml_file in yaml_files:
        try:
            with open(yaml_file, "r", encoding="utf-8") as f:
                content = f.read()
                if "BlackBoxSystem" in content or "GreyBoxSystem" in content:
                    system_files.append(yaml_file)
        except:
            continue

    print(f"📁 Found {len(system_files)} system configuration files")
    print()

    # Update each system file
    for file_path in system_files:
        if update_system_config(file_path):
            updated_count += 1

    print()
    print("=" * 60)
    print(f"🎉 Migration complete!")
    print(f"   📊 Files processed: {len(system_files)}")
    print(f"   ✅ Files updated: {updated_count}")
    print(f"   ⏭️  Files skipped: {len(system_files) - updated_count}")
    print()
    print("📝 Next steps:")
    print("   1. Test a few configurations to ensure they work")
    print("   2. Remove backup files (.yaml.backup) once confirmed")
    print("   3. Update any custom scripts that reference log_media_every_n_steps")


if __name__ == "__main__":
    main()
