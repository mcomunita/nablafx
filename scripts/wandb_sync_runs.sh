#!/bin/bash
# sync all wandb runs in a folder

# Set the folder path (use current directory by default)
folder="logs-scratch-02/marshall-jvm410h-ch-od1/GB-PARAM-DIST-MLP"
entity="team-mcomunita-qmul"
project="nnlinafx-PARAM"

# Find all .wandb files in the folder recursively
find "$folder" -type f -name "*.wandb" | sort | while read file; do
    # Run your instruction/command here for each file
    echo "Processing file: $file"
    sleep 10
    wandb sync -p "$project" -e "$entity" --include-online --include-offline --include-synced --no-mark-synced "$file"
done
