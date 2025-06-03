#!/bin/bash
# setup.sh
# Scriptin çalıştırılması için gerekli tüm izin ve bağımlılıkları ayarlar

# Renk tanımları
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}===================================================${NC}"
echo -e "${BLUE}       EEG Analiz Kurulum Scripti Başlatılıyor      ${NC}"
echo -e "${BLUE}===================================================${NC}"

# Klasör yapısını kontrol et/oluştur
echo -e "\n${YELLOW}Klasör yapısı hazırlanıyor...${NC}"
mkdir -p output/features
mkdir -p output/models
mkdir -p output/reports
echo -e "${GREEN}✓ Klasör yapısı oluşturuldu${NC}"

# Scriptlere çalıştırma izni ver
echo -e "\n${YELLOW}Çalıştırma izinleri ayarlanıyor...${NC}"
chmod +x run_pipeline.sh
chmod +x scripts/*.py

# Çalıştırma izinlerini kontrol et
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Çalıştırma izni ayarlanamadı!${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Çalıştırma izinleri ayarlandı${NC}"

# Python bağımlılıklarını kontrol et
echo -e "\n${YELLOW}Python bağımlılıkları kontrol ediliyor...${NC}"

# Python kurulumunu kontrol et
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 bulunamadı! Lütfen yükleyin.${NC}"
    echo -e "   İndirme: https://www.python.org/downloads/"
    exit 1
fi

# pip kurulumunu kontrol et
if ! command -v pip3 &> /dev/null; then
    echo -e "${RED}❌ pip3 bulunamadı! Lütfen yükleyin.${NC}"
    echo -e "   Python ile birlikte gelmiş olmalı, yoksa yükleyin: python -m ensurepip"
    exit 1
fi

# Gerekli kütüphaneleri yükle
echo -e "\n${YELLOW}Gerekli Python kütüphaneleri yükleniyor...${NC}"
pip3 install numpy pandas scipy scikit-learn matplotlib mne joblib

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Python kütüphanelerinin yüklenmesinde hata oluştu!${NC}"
    echo -e "   Lütfen manuel olarak yüklemeyi deneyin: pip3 install numpy pandas scipy scikit-learn matplotlib mne joblib"
    exit 1
fi
echo -e "${GREEN}✓ Python kütüphaneleri yüklendi${NC}"

# Veri seti kontrolü
echo -e "\n${YELLOW}CHB-MIT veri seti kontrol ediliyor...${NC}"
if [ ! -d "data/physionet.org/files/chbmit/1.0.0" ]; then
    echo -e "${YELLOW}⚠️ CHB-MIT veri seti bulunamadı!${NC}"
    echo -e "   Veri setini indirmek için: https://physionet.org/content/chbmit/1.0.0/"
    echo -e "   Veri setini 'data/physionet.org/files/chbmit/1.0.0' dizinine yerleştirin"
else
    echo -e "${GREEN}✓ CHB-MIT veri seti bulundu${NC}"
    
    # Kaç hasta var?
    PATIENT_COUNT=$(ls -d data/physionet.org/files/chbmit/1.0.0/chb* | wc -l)
    echo -e "   Toplam $PATIENT_COUNT hasta verisi mevcut"
fi

echo -e "\n${GREEN}===================================================${NC}"
echo -e "${GREEN}   ✓ Kurulum tamamlandı! Pipeline çalıştırılmaya hazır ${NC}"
echo -e "${GREEN}===================================================${NC}"
echo -e "\nKullanım:"
echo -e "  ${BLUE}./run_pipeline.sh${NC}                       # Tüm veri setini işle"
echo -e "  ${BLUE}./run_pipeline.sh --patient chb01${NC}       # Sadece chb01 hastasını işle"
echo -e "  ${BLUE}./run_pipeline.sh --file [DOSYA_YOLU]${NC}   # Tek bir EDF dosyasını işle"
echo -e "  ${BLUE}./run_pipeline.sh --help${NC}                # Yardım mesajını göster" 