#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
format_for_jmp.py

Bu script:
 1. Dengeli veri seti CSV dosyasını okur (varsayılan: output/balanced_dataset.csv)
 2. Kolon isimlerindeki '-' ve boşlukları '_' ile değiştirir
 3. label sütununu 0->'NonSeizure', 1->'Seizure' olarak kategorik yapar
 4. patient ve file sütunlarını string (nominal) tipine dönüştürür
 5. Sonucu JMP için hazırlanmış olarak kaydeder (varsayılan: output/balanced_dataset_jmp.csv)
"""

import os
import sys
import argparse
import pandas as pd

def format_for_jmp(input_path, output_path):
    """
    Veri setini JMP için hazırlar
    
    Parameters
    ----------
    input_path : str
        Okunacak CSV dosyası yolu
    output_path : str
        Yazılacak CSV dosyası yolu
    """
    # Dosya var mı kontrol et
    if not os.path.exists(input_path):
        print(f"❌ Hata: Dosya bulunamadı: {input_path}")
        return False
        
    try:
    # 1) CSV'i oku
        print(f"📂 Okunuyor: {input_path}")
    df = pd.read_csv(input_path)
        
        print(f"📊 Orijinal veri seti: {df.shape[0]} satır, {df.shape[1]} kolon")

    # 2) Kolon isimlerindeki '-' ve boşlukları temizle
    clean_cols = []
    for c in df.columns:
        # tire ve boşluk yerine alt çizgi
        new_c = c.replace('-', '_').replace(' ', '_')
            if new_c != c:
                print(f"  ✓ '{c}' -> '{new_c}'")
        clean_cols.append(new_c)
    df.columns = clean_cols

    # 3) label sütununu nominal (kategori) yap
    if 'label' not in df.columns:
            print("❌ Hata: Beklenen 'label' kolonu bulunamadı.")
            return False
            
        # label kolonunda 0 ve 1 dışında değer var mı kontrol et
        invalid_labels = set(df['label'].unique()) - {0, 1}
        if invalid_labels:
            print(f"❌ Hata: 'label' kolonunda beklenmeyen değerler: {invalid_labels}")
            return False
            
        # label değerlerini değiştir
    df['label'] = df['label'].map({0: 'NonSeizure', 1: 'Seizure'})
        print(f"  ✓ label sütunu kategorik hale getirildi: {df['label'].value_counts().to_dict()}")

    # 4) patient ve file sütunlarını string (nominal) tipine dönüştür
    for col in ('patient', 'file'):
        if col in df.columns:
            df[col] = df[col].astype(str)
                unique_values = len(df[col].unique())
                print(f"  ✓ '{col}' string tipine dönüştürüldü ({unique_values} benzersiz değer)")
        else:
            print(f"⚠️  Uyarı: '{col}' kolonu bulunamadı, atlanıyor.")

    # 5) Yeni dosyayı kaydet
    df.to_csv(output_path, index=False)
        print(f"✨ JMP için formatlanmış veri seti kaydedildi: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Hata: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Dengeli veri setini JMP istatistiksel analiz yazılımı için hazırlar")
    parser.add_argument("--input", type=str, 
                        default=os.path.join("output", "balanced_dataset.csv"),
                        help="Okunacak CSV dosyası (varsayılan: output/balanced_dataset.csv)")
    parser.add_argument("--output", type=str, 
                        default=os.path.join("output", "balanced_dataset_jmp.csv"),
                        help="Yazılacak CSV dosyası (varsayılan: output/balanced_dataset_jmp.csv)")
    
    args = parser.parse_args()
    
    success = format_for_jmp(args.input, args.output)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
