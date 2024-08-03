#!/bin/bash

SOURCE_FOLDER="output_wavs"
TARGET_FOLDER1="output_wavs_test_fake"
TARGET_FOLDER2="output_wavs_train_fake"

FILES=("$SOURCE_FOLDER"/*)

TOTAL_FILES=${#FILES[@]}
HALF_COUNT=$((TOTAL_FILES / 2))

for ((i=0; i<HALF_COUNT; i++)); do
    cp "${FILES[$i]}" "$TARGET_FOLDER1"
done

for ((i=HALF_COUNT; i<TOTAL_FILES; i++)); do
    cp "${FILES[$i]}" "$TARGET_FOLDER2"
done

echo "Success for allocate $TARGET_FOLDER1 and $TARGET_FOLDER2 !"
