#!/usr/bin/env python3
"""
Script to migrate all YAML configuration files from TimeAndFrequencyDomainLoss 
to the new WeightedMultiLoss system.

This script will:
1. Find all YAML files in cfg-new/ containing TimeAndFrequencyDomainLoss
2. Replace the loss configuration with WeightedMultiLoss equivalent
3. Preserve all existing parameters and structure
4. Create backups of original files
"""

import os
import re
import shutil
from pathlib import Path


def backup_file(file_path):
    """Create a backup of the original file."""
    backup_path = str(file_path) + '.backup'
    shutil.copy2(file_path, backup_path)
    print(f"Created backup: {backup_path}")


def migrate_loss_config(content):
    """
    Replace TimeAndFrequencyDomainLoss configuration with WeightedMultiLoss.
    
    Converts from:
        loss:
          class_path: nablafx.loss.TimeAndFrequencyDomainLoss
          init_args:
            time_domain_weight: .5
            frequency_domain_weight: .5

            time_domain_loss:
              class_path: torch.nn.L1Loss

            frequency_domain_loss:
              class_path: auraloss.freq.MultiResolutionSTFTLoss
    
    To:
        loss:
          class_path: nablafx.loss.WeightedMultiLoss
          init_args:
            losses:
              - loss:
                  class_path: torch.nn.L1Loss
                weight: 0.5
                name: "l1"
              - loss:
                  class_path: auraloss.freq.MultiResolutionSTFTLoss
                weight: 0.5
                name: "mrstft"
    """
    
    # Pattern to match the old loss configuration (with or without blank lines)
    old_pattern = re.compile(
        r'(\s*)loss:\s*\n'
        r'\1\s+class_path:\s*nablafx\.loss\.TimeAndFrequencyDomainLoss\s*\n'
        r'\1\s+init_args:\s*\n'
        r'\1\s+\s+time_domain_weight:\s*([\d.]+)\s*\n'
        r'\1\s+\s+frequency_domain_weight:\s*([\d.]+)\s*\n'
        r'(?:\s*\n)?'  # Optional blank line
        r'\1\s+\s+time_domain_loss:\s*\n'
        r'\1\s+\s+\s+class_path:\s*torch\.nn\.L1Loss\s*\n'
        r'(?:\s*\n)?'  # Optional blank line
        r'\1\s+\s+frequency_domain_loss:\s*\n'
        r'\1\s+\s+\s+class_path:\s*auraloss\.freq\.MultiResolutionSTFTLoss\s*',
        re.MULTILINE
    )
    
    def replace_loss(match):
        indent = match.group(1)
        time_weight = match.group(2)
        freq_weight = match.group(3)
        
        # Build the new WeightedMultiLoss configuration
        new_config = f"""{indent}loss:
{indent}  class_path: nablafx.loss.WeightedMultiLoss
{indent}  init_args:
{indent}    losses:
{indent}      - loss:
{indent}          class_path: torch.nn.L1Loss
{indent}        weight: {time_weight}
{indent}        name: "l1"
{indent}      - loss:
{indent}          class_path: auraloss.freq.MultiResolutionSTFTLoss
{indent}        weight: {freq_weight}
{indent}        name: "mrstft" """
        
        return new_config
    
    # Apply the replacement
    new_content = old_pattern.sub(replace_loss, content)
    
    return new_content


def process_file(file_path):
    """Process a single YAML file."""
    print(f"Processing: {file_path}")
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if it contains the old loss configuration
    if 'nablafx.loss.TimeAndFrequencyDomainLoss' not in content:
        print(f"  Skipping {file_path} - no TimeAndFrequencyDomainLoss found")
        return False
    
    # Create backup
    backup_file(file_path)
    
    # Migrate the content
    new_content = migrate_loss_config(content)
    
    # Write the updated content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"  ✅ Updated {file_path}")
    return True


def find_yaml_files(root_dir):
    """Find all YAML files in the directory tree."""
    yaml_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(('.yaml', '.yml')):
                yaml_files.append(Path(root) / file)
    return yaml_files


def main():
    """Main migration function."""
    # Configuration directory
    cfg_dir = Path('/homes/mc309/nablafx/cfg-new')
    
    if not cfg_dir.exists():
        print(f"Error: Configuration directory {cfg_dir} does not exist!")
        return
    
    print(f"🔍 Searching for YAML files in {cfg_dir}")
    
    # Find all YAML files
    yaml_files = find_yaml_files(cfg_dir)
    print(f"Found {len(yaml_files)} YAML files")
    
    # Process each file
    updated_count = 0
    for file_path in yaml_files:
        if process_file(file_path):
            updated_count += 1
    
    print(f"\n✅ Migration completed!")
    print(f"📊 Updated {updated_count} out of {len(yaml_files)} files")
    print(f"💾 Backup files created with .backup extension")
    
    # Show summary of what was changed
    if updated_count > 0:
        print(f"\n🔧 Changes made:")
        print(f"  - Replaced nablafx.loss.TimeAndFrequencyDomainLoss")
        print(f"  - With nablafx.loss.WeightedMultiLoss")  
        print(f"  - Preserved time_domain_weight and frequency_domain_weight")
        print(f"  - Added descriptive loss names: 'time_domain' and 'frequency_domain'")


if __name__ == '__main__':
    main()
