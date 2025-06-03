#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
normalize_features.py

Bu script:
 1. Özellik matrisini okur
 2. Farklı normalizasyon/standardizasyon yöntemleri uygular
 3. Normalize edilmiş veri setini kaydeder
"""

import os
import pandas as pd
import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

def standardize_features(df, method='standard', target_col='label', 
                        exclude_cols=None, save_scaler=True, plot=False):
    """
    Özellikleri normalize eder
    
    Parameters
    ----------
    df : pd.DataFrame
        Özellik matrisi
    method : str
        Normalizasyon metodu: 'standard', 'minmax', 'robust', 'yeo-johnson', 'quantile'
    target_col : str
        Hedef/etiket sütun adı
    exclude_cols : list or None
        Dönüşüm uygulanmayacak sütunlar
    save_scaler : bool
        Scaler nesnesini kaydetme
    plot : bool
        Normalizasyon öncesi/sonrası dağılımı görselleştirme
        
    Returns
    -------
    pd.DataFrame
        Normalize edilmiş özellikler
    """
    if exclude_cols is None:
        exclude_cols = []
    
    # Hedef değişkeni dönüşüm dışında tut
    if target_col not in exclude_cols:
        exclude_cols.append(target_col)
    
    # Kategorik değişkenleri belirle
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    exclude_cols.extend(cat_cols)
    
    # Dönüştürülecek sütunları belirle
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    if not feature_cols:
        print("⚠️ Uyarı: Dönüştürülecek sayısal sütun bulunamadı!")
        return df
    
    # İlk birkaç sütunun dağılımını göster (normalizasyon öncesi)
    if plot:
        plot_sample = min(5, len(feature_cols))
        plt.figure(figsize=(15, 5))
        for i, col in enumerate(feature_cols[:plot_sample]):
            plt.subplot(1, plot_sample, i+1)
            plt.hist(df[col], bins=30, alpha=0.7)
            plt.title(f"{col} (Orijinal)")
        plt.tight_layout()
        plt.savefig("output/pre_normalization.png", dpi=150)
        print("✓ Normalizasyon öncesi dağılım kaydedildi: output/pre_normalization.png")
    
    # Dönüşüm metodunu seç
    if method == 'minmax':
        scaler = MinMaxScaler()
        print("ℹ️ MinMax ölçekleme uygulanıyor (0 ile 1 arasında)")
    elif method == 'robust':
        scaler = RobustScaler()
        print("ℹ️ Robust ölçekleme uygulanıyor (median=0, IQR=1)")
    elif method == 'yeo-johnson':
        scaler = PowerTransformer(method='yeo-johnson')
        print("ℹ️ Yeo-Johnson dönüşümü uygulanıyor (normal dağılıma yaklaştırma)")
    elif method == 'quantile':
        scaler = QuantileTransformer(output_distribution='normal')
        print("ℹ️ Quantile dönüşümü uygulanıyor (uniform->normal)")
    else:  # Varsayılan: standard
        scaler = StandardScaler()
        print("ℹ️ Standart ölçekleme uygulanıyor (ortalama=0, std=1)")
    
    # Dönüşümü uygula
    normalized = df.copy()
    normalized[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # İlk birkaç sütunun dağılımını göster (normalizasyon sonrası)
    if plot:
        plt.figure(figsize=(15, 5))
        for i, col in enumerate(feature_cols[:plot_sample]):
            plt.subplot(1, plot_sample, i+1)
            plt.hist(normalized[col], bins=30, alpha=0.7)
            plt.title(f"{col} ({method})")
        plt.tight_layout()
        plt.savefig(f"output/post_normalization_{method}.png", dpi=150)
        print(f"✓ Normalizasyon sonrası dağılım kaydedildi: output/post_normalization_{method}.png")
    
    # Scaler'ı kaydet
    if save_scaler:
        scaler_path = f"output/scaler_{method}.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump((scaler, feature_cols), f)
        print(f"✓ Scaler kaydedildi: {scaler_path}")
    
    print(f"ℹ️ {len(feature_cols)} sayısal özellik normalize edildi.")
    
    return normalized

def main():
    parser = argparse.ArgumentParser(
        description="EEG özellik matrisi normalizasyonu yapar")
    parser.add_argument("--input", type=str, 
                        default="output/balanced_dataset.csv",
                        help="Giriş CSV dosyası (varsayılan: output/balanced_dataset.csv)")
    parser.add_argument("--output", type=str, 
                        default="output/normalized_features.csv",
                        help="Çıkış CSV dosyası (varsayılan: output/normalized_features.csv)")
    parser.add_argument("--method", type=str, 
                        choices=['standard', 'minmax', 'robust', 'yeo-johnson', 'quantile'],
                        default='standard',
                        help="Normalizasyon metodu (varsayılan: standard)")
    parser.add_argument("--exclude", type=str, nargs='+', default=[],
                        help="Dönüşüm dışında tutulacak sütun isimleri")
    parser.add_argument("--no-save-scaler", action="store_false", dest="save_scaler",
                        help="Scaler nesnesini kaydetme")
    parser.add_argument("--plot", action="store_true",
                        help="Normalizasyon öncesi/sonrası dağılımları görselleştir")
    
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
    
    # Normalizasyon
    normalized_df = standardize_features(
        df, method=args.method, 
        exclude_cols=args.exclude,
        save_scaler=args.save_scaler,
        plot=args.plot
    )
    
    # Sonuçları kaydet
    normalized_df.to_csv(args.output, index=False)
    print(f"✨ Normalize edilmiş veri seti kaydedildi: {args.output}")
    
    return 0

if __name__ == "__main__":
    main() 