#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
combine_features.py

Bu script:
 1. output/features/ klasöründeki tüm CSV dosyalarını bulur
 2. Hepsini tek bir CSV dosyasında birleştirir
"""

import pandas as pd
import os
import glob
import argparse
import sys

def combine_feature_files(input_dir, output_file):
    """
    Belirtilen dizindeki tüm CSV dosyalarını birleştirir
    
    Parameters
    ----------
    input_dir : str
        CSV dosyalarının bulunduğu dizin
    output_file : str
        Birleştirilmiş dosyanın kaydedileceği yer
    """
    print(f"Feature dosyalarını birleştirme işlemi başlatılıyor...")
    print(f"Kaynak dizin: {input_dir}")
    print(f"Hedef dosya: {output_file}")
    
    # CSV dosyalarını bul
    files = glob.glob(os.path.join(input_dir, '*.csv'))
    print(f"Toplam {len(files)} CSV dosyası bulundu.")
    
    if len(files) == 0:
        print("Hata: Hiç CSV dosyası bulunamadı!")
        return False
    
    # Her CSV dosyasının sütun sayısını kontrol et
    print("CSV dosyaları analiz ediliyor...")
    column_counts = {}
    all_dfs = []
    
    for i, file in enumerate(files):
        try:
            # Sadece başlığı oku
            header = pd.read_csv(file, nrows=0)
            column_count = len(header.columns)
            
            if column_count not in column_counts:
                column_counts[column_count] = []
            
            column_counts[column_count].append(file)
            
        except Exception as e:
            print(f"Dosya analiz hatası: {file} - {e}")
    
    # Kolon sayılarını rapor et
    print("\nKolon sayısı analizi:")
    for count, files_list in column_counts.items():
        print(f"  {count} kolonlu dosya sayısı: {len(files_list)}")
    
    # En yaygın kolon sayısını bul
    most_common_column_count = max(column_counts.items(), key=lambda x: len(x[1]))[0]
    print(f"\nEn yaygın kolon sayısı: {most_common_column_count}")
    print(f"Bu yapıya sahip {len(column_counts[most_common_column_count])} dosya kullanılacak")
    
    # Sadece uyumlu dosyaları birleştir
    compatible_files = column_counts[most_common_column_count]
    
    # İlk dosyanın başlık bilgisini al
    first = True
    total_rows = 0
    skipped_files = 0
    
    # Her dosyayı sırayla işle
    for i, file in enumerate(files):
        if file not in compatible_files:
            print(f"Atlanıyor: {i+1}/{len(files)} - {os.path.basename(file)} (kolon sayısı uyumsuz)")
            skipped_files += 1
            continue
            
        print(f"İşleniyor: {i+1}/{len(files)} - {os.path.basename(file)}")
        
        try:
            df = pd.read_csv(file)
            total_rows += len(df)
            
            # İlk dosya için başlıkları da ekle
            if first:
                df.to_csv(output_file, index=False)
                first = False
            else:
                # Diğer dosyalar için başlıkları ekleme
                df.to_csv(output_file, mode='a', header=False, index=False)
                
        except Exception as e:
            print(f"Hata: {file} - {e}")
            skipped_files += 1
    
    if total_rows == 0:
        print(f"Hata: Hiçbir dosya başarıyla işlenemedi!")
        return False
        
    print(f"\nTamamlandı! {len(files) - skipped_files} dosya birleştirildi ({skipped_files} dosya atlandı).")
    print(f"Toplam {total_rows} satır {output_file} dosyasına yazıldı.")
    return True

def main():
    parser = argparse.ArgumentParser(description="Feature dosyalarını birleştir")
    parser.add_argument("--input-dir", 
                      default="output/features", 
                      help="CSV dosyalarının bulunduğu dizin")
    parser.add_argument("--output", 
                      default="output/features_combined.csv", 
                      help="Birleştirilmiş dosya konumu")
    
    args = parser.parse_args()
    
    # Birleştirme işlemini çalıştır
    success = combine_feature_files(args.input_dir, args.output)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main() 