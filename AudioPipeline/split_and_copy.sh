#!/bin/bash

SOURCE_FOLDER="Chinese_Corpus"
TARGET_FOLDER1="Chinese_Corpus_Test_Real"
TARGET_FOLDER2="Chinese_Corpus_Train_Real"

FILES=("$SOURCE_FOLDER"/*)

TOTAL_FILES=${#FILES[@]}
HALF_COUNT=$((TOTAL_FILES / 2))

for ((i=0; i<HALF_COUNT; i++)); do
    cp "${FILES[$i]}" "$TARGET_FOLDER1"
done

for ((i=HALF_COUNT; i<TOTAL_FILES; i++)); do
    cp "${FILES[$i]}" "$TARGET_FOLDER2"
done

echo "文件已平均分配并复制到 $TARGET_FOLDER1 和 $TARGET_FOLDER2"
