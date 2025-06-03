#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import glob
import os
print("CSV dosyaları kontrol ediliyor...")
files = glob.glob("output/features/*.csv")
first_file = files[0]
print(f"İlk dosya: {first_file}")
first_df = pd.read_csv(first_file)
header = first_df.columns
column_count = len(header)
print(f"Referans sütun sayısı: {column_count}")
print("CSV dosyaları birleştiriliyor...")
all_dfs = []
skipped_files = []
for file in files:
    try:
        df = pd.read_csv(file)
        if len(df.columns) == column_count:
            all_dfs.append(df)
        else:
            print(f"Sütun sayısı uyumsuz: {file}, sütun sayısı: {len(df.columns)}")
            skipped_files.append(file)
    except Exception as e:
