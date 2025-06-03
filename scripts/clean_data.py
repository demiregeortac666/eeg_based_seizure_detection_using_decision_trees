#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
clean_data.py

Bu script:
 1. Özellik matrisini okur
 2. NaN (eksik) değerleri temizler
 3. Temizlenmiş veri setini kaydeder
"""

import os
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt

def clean_data(df, method='fill_zero', target_col='label', threshold=0.5, fill_value=0):
    """
    NaN değerleri temizler
    
    Parameters
    ----------
    df : pd.DataFrame
        Özellik matrisi
    method : str
        Temizleme metodu: 'fill_zero', 'fill_mean', 'fill_median', 'drop_rows', 'drop_cols'
    target_col : str
        Hedef/etiket sütun adı (her zaman korunur)
    threshold : float
        drop_cols için: bir sütundaki NaN değerlerin oranı bu değerden fazlaysa sütun silinir (0-1 arası)
    fill_value : int or float
        fill_value metodu için kullanılacak değer
        
    Returns
    -------
    pd.DataFrame
        Temizlenmiş veri seti
    """
    # Başlangıç bilgileri
    rows_before = df.shape[0]
    cols_before = df.shape[1]
    
    # Tüm özellikleri ve NaN istatistiklerini belirle
    nan_counts = df.isna().sum()
    total_nans = nan_counts.sum()
    
    print(f"⚠️ Toplam {total_nans} NaN değer bulundu ({total_nans/(rows_before*cols_before)*100:.2f}%)")
    
    # Kategorik ve hedef sütunlar
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    protected_cols = [target_col] if target_col in df.columns else []
    
    # Sayısal feature sütunlarını belirle (kategorik ve hedef hariç)
    feature_cols = [col for col in df.columns if col not in cat_cols + protected_cols]
    
    if method == 'fill_zero':
        print(f"ℹ️ NaN değerler 0 ile dolduruluyor...")
        df[feature_cols] = df[feature_cols].fillna(0)
    
    elif method == 'fill_mean':
        print(f"ℹ️ NaN değerler sütun ortalaması ile dolduruluyor...")
        for col in feature_cols:
            df[col] = df[col].fillna(df[col].mean())
    
    elif method == 'fill_median':
        print(f"ℹ️ NaN değerler sütun medyanı ile dolduruluyor...")
        for col in feature_cols:
            df[col] = df[col].fillna(df[col].median())
            
    elif method == 'fill_value':
        print(f"ℹ️ NaN değerler {fill_value} ile dolduruluyor...")
        df[feature_cols] = df[feature_cols].fillna(fill_value)
    
    elif method == 'drop_rows':
        print(f"ℹ️ NaN içeren satırlar siliniyor...")
        df = df.dropna()
        print(f"  Silinen satır sayısı: {rows_before - df.shape[0]}")
    
    elif method == 'drop_cols':
        # NaN oranı threshold'dan büyük olan sütunları belirle
        nan_ratio = nan_counts / rows_before
        drop_cols = [col for col in feature_cols if nan_ratio[col] > threshold 
                    and col not in protected_cols]
        
        if drop_cols:
            print(f"ℹ️ NaN oranı {threshold*100}%'dan fazla olan {len(drop_cols)} sütun siliniyor...")
            df = df.drop(columns=drop_cols)
            print(f"  Silinen sütunlar: {drop_cols[:5]}{'...' if len(drop_cols) > 5 else ''}")
        else:
            print(f"ℹ️ NaN oranı {threshold*100}%'dan fazla olan sütun bulunamadı.")
    
    # Sonuç bilgileri
    remaining_nans = df.isna().sum().sum()
    
    if remaining_nans > 0:
        print(f"⚠️ Temizleme sonrası {remaining_nans} NaN değer kaldı.")
    else:
        print(f"✓ Tüm NaN değerler temizlendi!")
    
    rows_after = df.shape[0]
    cols_after = df.shape[1]
    print(f"📊 Veri seti: {rows_before}x{cols_before} -> {rows_after}x{cols_after}")
    
    return df

def plot_nan_distribution(df, output_path="output/nan_distribution.png"):
    """NaN değerlerin dağılımını görselleştirir"""
    # NaN dağılımını hesapla
    nan_counts = df.isna().sum().sort_values(ascending=False)
    nan_ratio = (nan_counts / len(df) * 100).round(2)
    
    # NaN içeren ilk 20 sütunu göster
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) == 0:
        print("ℹ️ NaN değer bulunmadığından grafik oluşturulmadı.")
        return
    
    plot_cols = min(20, len(nan_cols))
    plt.figure(figsize=(12, 6))
    plt.bar(range(plot_cols), nan_ratio.iloc[:plot_cols], color='skyblue')
    plt.xticks(range(plot_cols), nan_ratio.index[:plot_cols], rotation=90)
    plt.ylabel('NaN Oranı (%)')
    plt.title('NaN Değer Dağılımı (İlk 20 Sütun)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"✓ NaN dağılım grafiği kaydedildi: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="EEG özellik matrisinden NaN değerleri temizler")
    parser.add_argument("--input", type=str, 
                        default="output/normalized_features.csv",
                        help="Giriş CSV dosyası (varsayılan: output/normalized_features.csv)")
    parser.add_argument("--output", type=str, 
                        default="output/cleaned_features.csv",
                        help="Çıkış CSV dosyası (varsayılan: output/cleaned_features.csv)")
    parser.add_argument("--method", type=str, 
                        choices=['fill_zero', 'fill_mean', 'fill_median', 'fill_value', 
                                'drop_rows', 'drop_cols'],
                        default='fill_zero',
                        help="Temizleme metodu (varsayılan: fill_zero)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="drop_cols için NaN oranı eşiği (0-1 arası, varsayılan: 0.5)")
    parser.add_argument("--fill-value", type=float, default=0,
                        help="fill_value metodu için kullanılacak değer (varsayılan: 0)")
    parser.add_argument("--plot", action="store_true",
                        help="NaN dağılımını görselleştir")
    
    args = parser.parse_args()
    
    # Giriş dosyasını kontrol et
    if not os.path.exists(args.input):
        print(f"❌ Hata: Dosya bulunamadı: {args.input}")
        return 1
    
    # Çıkış dizinini kontrol et
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"📂 Okunuyor: {args.input}")
    df = pd.read_csv(args.input)
    
    print(f"📊 Orijinal veri seti: {df.shape[0]} satır, {df.shape[1]} kolon")
    
    # NaN dağılımını görselleştir
    if args.plot:
        plot_nan_distribution(df)
    
    # Veriyi temizle
    cleaned_df = clean_data(
        df, method=args.method, 
        threshold=args.threshold,
        fill_value=args.fill_value
    )
    
    # Sonuçları kaydet
    cleaned_df.to_csv(args.output, index=False)
    print(f"✨ Temizlenmiş veri seti kaydedildi: {args.output}")
    
    return 0

if __name__ == "__main__":
    main() 