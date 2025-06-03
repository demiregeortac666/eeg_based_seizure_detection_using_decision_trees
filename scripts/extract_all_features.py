#!/usr/bin/env python3
# scripts/extract_all_features.py

import os
import sys
import numpy as np
import pandas as pd
from read_eeg_and_summary import segment_eeg_file
from feature_extraction import extract_features

# Proje kök dizini için relatif pathlar
DATA_ROOT = "data/physionet.org/files/chbmit/1.0.0"
OUTPUT_DIR = "output/features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def check_segments(segs, lbls, chs, fs):
    """Segment verilerinin geçerli olup olmadığını kontrol eder"""
    if len(segs) == 0:
        return False, "Boş segment dizisi"
    
    if fs <= 0:
        return False, "Geçersiz örnekleme frekansı"
    
    if len(chs) == 0:
        return False, "Kanal isimleri bulunamadı"
    
    if len(lbls) != len(segs):
        return False, f"Segment ({len(segs)}) ve etiket ({len(lbls)}) boyutları eşleşmiyor"
    
    return True, "OK"

def process_files(patients=None):
    """
    CHB-MIT veri setindeki EDF dosyalarından özellik çıkarır
    
    Parameters
    ----------
    patients : list or None
        İşlenecek hasta ID'leri listesi (None: tüm hastalar)
    """
    all_dfs = []
    processed_count = 0
    error_count = 0
    skipped_count = 0
    
    # Tüm hastaları veya belirtilen hastaları listele
    patient_dirs = []
    if patients:
        for p in patients:
            pdir = os.path.join(DATA_ROOT, p)
            if os.path.isdir(pdir):
                patient_dirs.append((p, pdir))
            else:
                print(f"⚠️ Hasta dizini bulunamadı: {p}")
    else:
        for p in sorted(os.listdir(DATA_ROOT)):
            pdir = os.path.join(DATA_ROOT, p)
            if os.path.isdir(pdir) and p.startswith('chb'):
                patient_dirs.append((p, pdir))
    
    # İşlenecek hasta sayısını göster
    print(f"📊 Toplam {len(patient_dirs)} hasta için işlem yapılacak")
    
    # Her hasta için işlem yap
    for patient, pdir in patient_dirs:
        print(f"\n🏥 Hasta işleniyor: {patient}")
        
        patient_files = []
        for fname in sorted(os.listdir(pdir)):
            if fname.endswith(".edf"):
                patient_files.append(fname)
        
        print(f"   Toplam {len(patient_files)} EDF dosyası bulundu")
        
        patient_processed = 0
        patient_errors = 0
        
        for fname in patient_files:
            edf_path = os.path.join(pdir, fname)
seizure_path = edf_path + ".seizures"

            print(f"\n⌛ İşleniyor: {edf_path}")
            
            # segment & extract features
            try:
                segs, lbls, chs, fs = segment_eeg_file(edf_path, seizure_path)
                
                # Segment verilerinin geçerli olup olmadığını kontrol et
                valid, reason = check_segments(segs, lbls, chs, fs)
                if not valid:
                    print(f"⚠️ Geçersiz segment: {edf_path} - {reason}")
                    skipped_count += 1
                    continue
                
                # Özellikleri çıkar
                df = extract_features(segs, lbls, chs, fs)
                
                # Metadata ekle
                df["patient"] = patient
                df["file"] = fname
                
                # Birleşik veri setine ekle
                all_dfs.append(df)

                # Dosyayı ayrı kaydet
                outp = os.path.join(OUTPUT_DIR, f"feat_{patient}_{fname}.csv")
                df.to_csv(outp, index=False)
                print(f"✅ Kaydedildi: {outp}")
                print(f"   {len(df)} örnek, {df.shape[1]} özellik")
                print(f"   Sınıf dağılımı: {df['label'].value_counts().to_dict()}")
                
                patient_processed += 1
                processed_count += 1
                
            except Exception as e:
                print(f"❌ Hata: {edf_path} - {str(e)}")
                import traceback
                traceback.print_exc()
                patient_errors += 1
                error_count += 1
        
        print(f"\n📈 Hasta {patient} özeti:")
        print(f"   İşlenen: {patient_processed}/{len(patient_files)} dosya")
        print(f"   Hata: {patient_errors} dosya")

    # İşlem sonuçlarını göster
    print("\n🔍 İşlem Sonuçları:")
    print(f"   Toplam işlenen: {processed_count} dosya")
    print(f"   Atlanan: {skipped_count} dosya")
    print(f"   Hatalı: {error_count} dosya")

    # Tümünü birleştir
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_path = "output/features_combined.csv"
        combined.to_csv(combined_path, index=False)
        print(f"\n✨ Birleştirilmiş özellikler kaydedildi: {combined_path}")
        print(f"   Toplam örnek sayısı: {len(combined)}")
        print(f"   Toplam özellik sayısı: {combined.shape[1]}")
        print(f"   Sınıf dağılımı: {combined['label'].value_counts().to_dict()}")
        return True
    else:
        print("\n❌ Hiç geçerli veri işlenemedi, birleştirilmiş dosya oluşturulamadı.")
        return False

def process_single_file(edf_path):
    """Tek bir EDF dosyasından özellik çıkarır"""
    if not os.path.exists(edf_path):
        print(f"❌ Hata: Dosya bulunamadı: {edf_path}")
        return False
        
    seizure_path = edf_path + ".seizures"
    output_file = os.path.basename(edf_path)
    output_csv = os.path.join(OUTPUT_DIR, f"feat_{output_file}.csv")
    
    print(f"⌛ Dosya işleniyor: {edf_path}")
    
    try:
# Segment and extract
segments, labels, ch_names, fs = segment_eeg_file(edf_path, seizure_path)
        
        # Segment verilerinin geçerli olup olmadığını kontrol et
        valid, reason = check_segments(segments, labels, ch_names, fs)
        if not valid:
            print(f"⚠️ Geçersiz segment: {edf_path} - {reason}")
            return False
        
        # Özellikleri çıkar
        features_df = extract_features(segments, labels, ch_names, fs)
        
        # Dosya bilgisini ekle
        features_df["file"] = os.path.basename(edf_path)

# Save
features_df.to_csv(output_csv, index=False)
        print(f"✅ Özellik matrisi kaydedildi: {output_csv}")
        print(f"   Boyut: {features_df.shape}")
        print(f"   Sınıf dağılımı: {features_df['label'].value_counts().to_dict()}")
        return True
        
    except Exception as e:
        print(f"❌ Hata: {edf_path} - {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="EEG dosyalarından özellik çıkarma")
    parser.add_argument("--file", type=str, help="İşlenecek tek EDF dosyası (isteğe bağlı)")
    parser.add_argument("--patient", type=str, nargs='+', help="İşlenecek hasta ID'leri (chb01, chb02, ...)")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, 
                        help=f"Çıktı dizini (varsayılan: {OUTPUT_DIR})")
    
    args = parser.parse_args()
    
    # Çıktı dizinini ayarla
    if args.output_dir != OUTPUT_DIR:
        OUTPUT_DIR = args.output_dir
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"📁 Çıktı dizini: {OUTPUT_DIR}")
    
    # Tek dosya veya hasta listesi veya tüm veri seti işleme
    if args.file:
        success = process_single_file(args.file)
    else:
        success = process_files(args.patient)
    
    # Başarı durumuna göre çık
    sys.exit(0 if success else 1)
