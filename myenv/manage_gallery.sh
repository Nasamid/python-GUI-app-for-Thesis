#!/bin/bash
# Directory where your images and gallery_data.json are stored.
GALLERY_DIR="/home/jr/myenv/gallery"
JSON_FILE="$GALLERY_DIR/gallery_data.json"

# Number of entries to keep (20 entries = 20 raw + 20 processed images)
KEEP_ENTRIES=20

# Run indefinitely; you can also schedule this with cron if desired.
while true; do
    # Only process if the JSON file exists.
    if [ -f "$JSON_FILE" ]; then
        # Get number of entries in the JSON file.
        count=$(jq '. | length' "$JSON_FILE")
        echo "Gallery JSON entries: $count"
        
        # If the number of entries is greater than the number to keep, trim the JSON.
        if [ "$count" -gt "$KEEP_ENTRIES" ]; then
            echo "Trimming gallery JSON to last $KEEP_ENTRIES entries..."
            # Backup the current JSON.
            cp "$JSON_FILE" "$JSON_FILE.bak"
            
            # Keep only the first KEEP_ENTRIES elements.
            jq ".[0:$KEEP_ENTRIES]" "$JSON_FILE" > "$JSON_FILE.tmp" && mv "$JSON_FILE.tmp" "$JSON_FILE"
            
            # Extract list of raw and processed image filenames to keep.
            keep_raw=$(jq -r '.[].raw_image' "$JSON_FILE" | xargs -n1 basename)
            keep_proc=$(jq -r '.[].processed_image' "$JSON_FILE" | xargs -n1 basename)
            
            echo "Raw images to keep:"
            echo "$keep_raw"
            echo "Processed images to keep:"
            echo "$keep_proc"
            
            # Remove raw image files in GALLERY_DIR that are not in the keep list.
            for file in "$GALLERY_DIR"/captured_image_*.jpg; do
                filename=$(basename "$file")
                if ! echo "$keep_raw" | grep -qx "$filename"; then
                    echo "Deleting raw image: $filename"
                    rm "$file"
                fi
            done
            
            # Remove processed image files in GALLERY_DIR that are not in the keep list.
            for file in "$GALLERY_DIR"/processed_captured_image_*.png; do
                filename=$(basename "$file")
                if ! echo "$keep_proc" | grep -qx "$filename"; then
                    echo "Deleting processed image: $filename"
                    rm "$file"
                fi
            done
        fi
    fi
    # Wait for 60 seconds before next check.
    sleep 60
done
