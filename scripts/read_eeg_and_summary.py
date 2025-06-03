# scripts/read_eeg_and_summary.py

import os
import numpy as np
import mne
import warnings
import re

def load_seizure_annotations(seizure_path, edf_path=None, verbose=True):
    """
    Nöbet aralıklarını yükler. Hem .seizures dosyalarından hem de 
    summary.txt dosyasından okumayı dener.
    
    Parameters
    ----------
    seizure_path : str
        .seizures dosyasının yolu
    edf_path : str, optional
        EDF dosyasının yolu, summary.txt'den okuma için gerekli
    verbose : bool
        Ayrıntılı çıktı göster/gizle
    
    Returns
    -------
    list
        (onset, offset) tuple'larından oluşan nöbet aralıkları listesi
    """
    seizure_intervals = []
    
    # 1. .seizures dosyasından okumayı dene
    try:
        if verbose:
            print(f"📂 Nöbet dosyası okunuyor: {seizure_path}")
            
        with open(seizure_path, "rb") as f:
            content = f.read().decode(errors="ignore")
            if verbose:
                print(f"📝 Dosya içeriği: {content.strip()}")
                
            lines = content.splitlines()
            for line in lines:
                # Boş satırları atla
                if not line.strip():
                    continue
                    
                # İki sayı biçiminde mi?
                parts = line.strip().split()
                if len(parts) == 2:
                    try:
                    onset, offset = float(parts[0]), float(parts[1])
                    seizure_intervals.append((onset, offset))
                        if verbose:
                            print(f"⏱️ Nöbet aralığı bulundu: {onset}s - {offset}s")
                    except ValueError:
                        print(f"⚠️ Geçersiz zaman değeri: {parts} in {seizure_path}")
    except FileNotFoundError:
        if verbose:
            print(f"⚠️ Nöbet dosyası bulunamadı: {seizure_path}")
    
    # 2. Eğer nöbet aralığı bulunamadıysa ve edf_path verilmişse summary.txt'den okumayı dene
    if len(seizure_intervals) == 0 and edf_path is not None:
        try:
            # EDF dosya adını al
            edf_filename = os.path.basename(edf_path)
            patient_id = edf_filename.split('_')[0]  # örn. "chb01"
            
            # summary.txt yolunu oluştur
            summary_path = os.path.join(os.path.dirname(edf_path), f"{patient_id}-summary.txt")
            
            if os.path.exists(summary_path):
                if verbose:
                    print(f"📂 Nöbetler için özet dosyası okunuyor: {summary_path}")
                
                with open(summary_path, "r", errors="ignore") as f:
                    content = f.read()
                    
                    # Düzenli ifade ile dosya ve nöbet bilgilerini eşleştir
                    file_pattern = re.compile(r"File Name: (.*?)\n.*?Number of Seizures in File: (\d+)(.*?)(?=File Name:|$)", re.DOTALL)
                    seizure_pattern = re.compile(r"Seizure Start Time: (\d+) seconds\nSeizure End Time: (\d+) seconds")
                    
                    for match in file_pattern.finditer(content):
                        filename, seizure_count, seizure_info = match.groups()
                        
                        # Eğer bu bizim aranan dosya ise ve nöbetler varsa
                        if filename == edf_filename and int(seizure_count) > 0:
                            if verbose:
                                print(f"📊 {filename} için {seizure_count} nöbet bilgisi bulundu")
                                
                            # Nöbet başlangıç ve bitiş zamanlarını bul
                            for sz_match in seizure_pattern.finditer(seizure_info):
                                onset, offset = int(sz_match.group(1)), int(sz_match.group(2))
                                seizure_intervals.append((onset, offset))
                                if verbose:
                                    print(f"⏱️ Nöbet aralığı bulundu (summary.txt): {onset}s - {offset}s")
        except Exception as e:
            print(f"⚠️ Özet dosyası işlenirken hata: {str(e)}")
    
    if len(seizure_intervals) > 0:
        if verbose:
            print(f"✅ Toplam {len(seizure_intervals)} nöbet aralığı yüklendi")
    else:
        if verbose:
            print(f"ℹ️ Hiç nöbet aralığı bulunamadı")
            
    return seizure_intervals

def is_seizure_window(start_t, end_t, intervals):
    """
    Verilen zaman penceresinin herhangi bir nöbet aralığıyla örtüşüp örtüşmediğini kontrol eder
    
    Parameters
    ----------
    start_t : float
        Pencere başlangıç zamanı (saniye)
    end_t : float
        Pencere bitiş zamanı (saniye)
    intervals : list
        (onset, offset) tuple'larından oluşan nöbet aralıkları listesi
        
    Returns
    -------
    int
        Eğer pencere bir nöbet aralığıyla örtüşüyorsa 1, aksi halde 0
    """
    # Aralık kontrollerini yap
    window_duration = end_t - start_t
    overlap_threshold = 0.5  # Pencerenin en az %50'si nöbet olmalı
    
    for onset, offset in intervals:
        # Örtüşme miktarını hesapla
        overlap_start = max(start_t, onset)
        overlap_end = min(end_t, offset)
        
        if overlap_end > overlap_start:  # Örtüşme varsa
            overlap_duration = overlap_end - overlap_start
            overlap_ratio = overlap_duration / window_duration
            
            # Eğer yeterli örtüşme varsa, pencere bir nöbet olarak etiketlenir
            if overlap_ratio >= overlap_threshold:
            return 1
    
    return 0

def segment_eeg_file(edf_path, seizure_path,
                     window_sec=10, step_sec=5,
                     verbose=False):
    """
    EEG dosyasını segmentlere ayırır ve etiketler
    
    Parameters
    ----------
    edf_path : str
        EEG dosyasının yolu
    seizure_path : str
        Nöbet annotation dosyasının yolu
    window_sec : int
        Segment pencere uzunluğu (saniye)
    step_sec : int
        Segment adım uzunluğu (saniye)
    verbose : bool
        Ayrıntılı çıktı göster/gizle
        
    Returns
    -------
    segments : np.array
        (N_windows, N_channels, N_samples) şeklinde array
    labels : np.array
        (N_windows,) şeklinde 0/1 etiketleri
    ch_names : list
        Kanal isimleri
    sfreq : float
        Örnekleme frekansı
    """
    try:
        # Kanal adı çoğaltma uyarılarını bastır
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, 
                                   message="Channel names are not unique")
            
            # EDF dosyasını oku
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            
            # Sadece EEG kanallarını seç
    raw.pick_types(eeg=True)
            
            # Kanal isimlerindeki çoğaltmaları işle
            ch_names = raw.info['ch_names']
            ch_name_counts = {}
            for i, name in enumerate(ch_names):
                if name in ch_name_counts:
                    ch_name_counts[name] += 1
                    # Yinelenen kanala benzersiz bir isim ver
                    new_name = f"{name}_{ch_name_counts[name]}"
                    raw.rename_channels({name: new_name})
                else:
                    ch_name_counts[name] = 0
            
            # Average referans uygula
    raw.set_eeg_reference('average', projection=True)
    
            # Veriyi al
    data = raw.get_data()
    ch_names = raw.info['ch_names']
    sfreq = raw.info['sfreq']
    total_samples = data.shape[1]
            total_duration = total_samples / sfreq
            
            if verbose:
                print(f"📊 EEG veri boyutu: {data.shape}")
                print(f"⏱️ Toplam süre: {total_duration:.2f} saniye")

            # Nöbet annotation'larını yükle (hem .seizures hem summary.txt'den)
            intervals = load_seizure_annotations(seizure_path, edf_path, verbose=verbose)
            
            # Segment parametrelerini hesapla
    win_samp = int(window_sec * sfreq)
    step_samp = int(step_sec * sfreq)

            # Eğer toplam örneklem segmentasyona yeterli değilse, boş döndür
            if total_samples <= win_samp:
                print(f"⚠️ Dosya çok kısa, segmentasyon yapılamıyor: {edf_path}")
                print(f"   Toplam örneklem: {total_samples}, Gereken: {win_samp}")
                return np.array([]), np.array([]), [], 0

            # Segmentlere ayır
    segments, labels = [], []
            num_labeled_segments = 0
            
    for start in range(0, total_samples - win_samp + 1, step_samp):
        end = start + win_samp
        seg = data[:, start:end]
                
                # Saniye cinsinden zaman değerleri
                start_t = start / sfreq
                end_t = end / sfreq
                
                # Etiketleme
                lbl = is_seizure_window(start_t, end_t, intervals)
                if lbl == 1:
                    num_labeled_segments += 1
                
        segments.append(seg)
        labels.append(lbl)

            # Eğer hiç segment oluşmadıysa, boş döndür
            if len(segments) == 0:
                print(f"⚠️ Hiç segment oluşturulamadı: {edf_path}")
                return np.array([]), np.array([]), [], 0
                
            # Etiket sayısı kontrolü
            segments_array = np.array(segments)
            labels_array = np.array(labels)
            
            if verbose or num_labeled_segments > 0:
                print(f"📈 Toplam {len(segments)} segment, {num_labeled_segments} nöbet etiketi (%{(num_labeled_segments/len(segments)*100):.2f})")
                
            return segments_array, labels_array, ch_names, sfreq
            
    except Exception as e:
        print(f"⚠️ EEG dosyası işlenirken hata oluştu: {edf_path}")
        print(f"   Hata: {str(e)}")
        import traceback
        traceback.print_exc()
        return np.array([]), np.array([]), [], 0

# Test fonksiyonu - geliştirme sırasında kullanılır
def test_seizure_loading(edf_path):
    """
    Nöbet yükleme işlemini test eder
    """
    seizure_path = edf_path + ".seizures"
    intervals = load_seizure_annotations(seizure_path, edf_path, verbose=True)
    return intervals
