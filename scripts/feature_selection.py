#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
feature_selection.py

Bu script:
 1. Özellik matrisini okur
 2. Korelasyon tabanlı ve önem bazlı özellik seçimi yapar
 3. Seçilen özelliklerle daha küçük boyutlu veri seti oluşturur
"""

import os
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

def correlation_filter(df, threshold=0.9, target_col='label', plot=False):
    """
    Yüksek korelasyona sahip özellikleri filtreler
    
    Parameters
    ----------
    df : pd.DataFrame
        Özellik matrisi
    threshold : float
        Korelasyon eşiği (varsayılan: 0.9)
    target_col : str
        Hedef/etiket sütun adı
    plot : bool
        Korelasyon matrisini görselleştirme
        
    Returns
    -------
    pd.DataFrame
        Filtrelenmiş özellik matrisi
    """
    # Sadece sayısal özellikleri seç
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Hedef değişkeni hariç tut
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    # Korelasyon matrisi
    corr_matrix = df[numeric_cols].corr().abs()
    
    # Korelasyon matrisini görselleştir
    if plot:
        plt.figure(figsize=(12, 10))
        plt.title("Özellik Korelasyon Matrisi", fontsize=15)
        plt.imshow(corr_matrix, cmap='coolwarm')
        plt.colorbar()
        plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90)
        plt.yticks(range(len(numeric_cols)), numeric_cols)
        plt.tight_layout()
        plt.savefig("output/correlation_matrix.png", dpi=150)
        print("✓ Korelasyon matrisi kaydedildi: output/correlation_matrix.png")
    
    # Yüksek korelasyona sahip özellikleri belirle
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > threshold)]
    
    print(f"ℹ️ Yüksek korelasyonlu {len(to_drop)} özellik çıkarıldı.")
    print(f"  Eşik değeri: {threshold}")
    
    # Kalan özellikleri döndür
    return df.drop(columns=to_drop)

def feature_importance(df, target_col='label', n_features=30, plot=False):
    """
    Rastgele orman modeliyle özellik önemini hesaplar
    
    Parameters
    ----------
    df : pd.DataFrame
        Özellik matrisi
    target_col : str
        Hedef/etiket sütun adı
    n_features : int
        Seçilecek özellik sayısı
    plot : bool
        Özellik önemini görselleştirme
        
    Returns
    -------
    pd.DataFrame
        Önem sıralamasına göre filtrelenmiş özellikler
    """
    # Sayısal ve kategorik olmayan özellikleri seç
    exclude_cols = [col for col in df.columns if 
                     col == target_col or 
                     df[col].dtype == 'object' or 
                     df[col].nunique() < 5]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Özellikleri standartlaştır
    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Hedef değişkeni
    y = df[target_col].values
    
    # Rastgele orman modeli
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_scaled, y)
    
    # Özellik önemini hesapla
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Özellik önemini görselleştir
    if plot:
        plt.figure(figsize=(12, 8))
        plt.title("Özellik Önem Sıralaması", fontsize=15)
        plt.bar(range(min(30, len(feature_cols))), 
                importances[indices[:30]], color='royalblue')
        plt.xticks(range(min(30, len(feature_cols))), 
                   [feature_cols[i] for i in indices[:30]], rotation=90)
        plt.tight_layout()
        plt.savefig("output/feature_importance.png", dpi=150)
        print("✓ Özellik önem grafiği kaydedildi: output/feature_importance.png")
    
    # Özellik önemlerine göre DataFrame oluştur
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Özellik önem tablosunu dışa aktar
    importance_df.to_csv("output/feature_importance.csv", index=False)
    print(f"✓ Özellik önem tablosu kaydedildi: output/feature_importance.csv")
    
    # En önemli n özelliği seç
    top_features = [feature_cols[i] for i in indices[:n_features]]
    
    # Tüm değişkenleri (kategorik dahil) dahil et
    selected_cols = top_features + exclude_cols
    
    print(f"ℹ️ Önem sıralamasına göre {len(top_features)} özellik seçildi.")
    
    return df[selected_cols]

def univariate_selection(df, target_col='label', method='f_classif', n_features=30):
    """
    Tek değişkenli özellik seçimi
    
    Parameters
    ----------
    df : pd.DataFrame
        Özellik matrisi
    target_col : str
        Hedef/etiket sütun adı
    method : str
        Seçim metodu ('f_classif' veya 'mutual_info')
    n_features : int
        Seçilecek özellik sayısı
        
    Returns
    -------
    pd.DataFrame
        Seçilmiş özelliklerle DataFrame
    """
    # Hedef değişkeni ve kategorik değişkenleri hariç tut
    exclude_cols = [col for col in df.columns if 
                    col == target_col or 
                    df[col].dtype == 'object' or 
                    df[col].nunique() < 5]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Özellikleri standartlaştır
    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Hedef değişkeni
    y = df[target_col].values
    
    # Seçim metodu
    if method == 'mutual_info':
        selector = SelectKBest(mutual_info_classif, k=n_features)
    else:  # Varsayılan: f_classif
        selector = SelectKBest(f_classif, k=n_features)
    
    # Özellik seçimi
    selector.fit(X_scaled, y)
    
    # Seçilen özelliklerin indexleri
    selected_indices = selector.get_support(indices=True)
    selected_features = [feature_cols[i] for i in selected_indices]
    
    # Tüm değişkenleri (kategorik dahil) dahil et
    selected_cols = selected_features + exclude_cols
    
    print(f"ℹ️ {method} metodu ile {len(selected_features)} özellik seçildi.")
    
    return df[selected_cols]

def main():
    parser = argparse.ArgumentParser(
        description="EEG özellik matrisi için özellik seçimi yapar")
    parser.add_argument("--input", type=str, 
                        default="output/balanced_dataset.csv",
                        help="Giriş CSV dosyası (varsayılan: output/balanced_dataset.csv)")
    parser.add_argument("--output", type=str, 
                        default="output/selected_features.csv",
                        help="Çıkış CSV dosyası (varsayılan: output/selected_features.csv)")
    parser.add_argument("--method", type=str, choices=['correlation', 'importance', 'f_classif', 'mutual_info'],
                        default='importance',
                        help="Özellik seçim metodu (varsayılan: importance)")
    parser.add_argument("--n_features", type=int, default=30,
                        help="Seçilecek özellik sayısı (varsayılan: 30)")
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="Korelasyon eşiği (correlation metodu için) (varsayılan: 0.9)")
    parser.add_argument("--plot", action="store_true",
                        help="Görselleştirme yap")
    
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
    
    # Özellik seçimi
    if args.method == 'correlation':
        selected_df = correlation_filter(df, threshold=args.threshold, plot=args.plot)
    elif args.method == 'importance':
        selected_df = feature_importance(df, n_features=args.n_features, plot=args.plot)
    elif args.method == 'f_classif':
        selected_df = univariate_selection(df, method='f_classif', n_features=args.n_features)
    elif args.method == 'mutual_info':
        selected_df = univariate_selection(df, method='mutual_info', n_features=args.n_features)
    
    # Sonuçları kaydet
    selected_df.to_csv(args.output, index=False)
    print(f"✨ Seçilen özelliklerle veri seti kaydedildi: {args.output}")
    print(f"  Yeni boyut: {selected_df.shape[0]} satır, {selected_df.shape[1]} kolon")
    
    return 0

if __name__ == "__main__":
    main() 