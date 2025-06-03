#!/usr/bin/env python3
# scripts/create_balanced_dataset.py

import os
import argparse
import pandas as pd

def create_balanced_dataset(input_file, output_file, pos_ratio=2):
    """
    Pozitif ve negatif örnekleri dengeleyerek yeni bir veri seti oluşturur
    
    Parameters
    ----------
    input_file : str
        Giriş CSV dosyası
    output_file : str
        Çıkış CSV dosyası
    pos_ratio : int, default=2
        Negatif/Pozitif örnek oranı (2 = 2 kat negatif örnek)
    """
    print(f"Okunuyor: {input_file}")
    df = pd.read_csv(input_file)
    
# temel kontrol
    if "label" not in df.columns:
        raise ValueError("Veri setinde 'label' sütunu bulunamadı!")
    
    # Sınıfları ayır
    seizure_samples = df[df.label==1]
    non_seizure_samples = df[df.label==0]
    
    print(f"Orijinal veri seti:")
    print(f"  Toplam örnekler: {len(df)}")
    print(f"  Nöbet örnekleri: {len(seizure_samples)}")
    print(f"  Normal örnekler: {len(non_seizure_samples)}")
    
    if len(seizure_samples) == 0:
        raise ValueError("Veri setinde nöbet örneği bulunamadı!")
    
    # Negatif örnekleri seç
    n_neg_samples = min(len(non_seizure_samples), len(seizure_samples) * pos_ratio)
    selected_non_seizure = non_seizure_samples.sample(
        n=n_neg_samples, random_state=42)
    
    # Veri setini oluştur ve karıştır
    balanced = pd.concat([seizure_samples, selected_non_seizure], 
                         ignore_index=True).sample(frac=1, random_state=42)
    
    # Kaydet
    balanced.to_csv(output_file, index=False)
    
    print("\n✅ Dengelenmiş veri seti kaydedildi:")
    print(f"  Dosya: {output_file}")
    print(f"  Toplam örnekler: {len(balanced)}")
    print(f"  Sınıf dağılımı:")
    print(balanced.label.value_counts())
    
    return balanced

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EEG veri seti için dengeli bir veri seti oluşturur")
    parser.add_argument("--input", type=str, 
                        default="output/features_combined.csv",
                        help="Giriş özellik CSV dosyası")
    parser.add_argument("--output", type=str, 
                        default="output/balanced_dataset.csv",
                        help="Çıkış CSV dosyası")
    parser.add_argument("--ratio", type=int, default=2,
                        help="Negatif/Pozitif örnek oranı (varsayılan: 2)")
    
    args = parser.parse_args()
    
    create_balanced_dataset(args.input, args.output, args.ratio)
