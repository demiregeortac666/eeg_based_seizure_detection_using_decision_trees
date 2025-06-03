#!/bin/bash
# run_from_features.sh
#
# Bu script features dizinindeki EEG özelliklerini kullanarak analiz iş akışındaki adımları sırayla çalıştırır:
# 1. Feature dosyalarını birleştirme
# 2. Dengeli veri seti oluşturma
# 3. Özellik seçimi
# 4. Normalizasyon
# 5. NaN Değerlerini Temizleme
# 6. Cross-validation ve model eğitimi
# 7. JMP formatına dönüştürme

set -e  # Herhangi bir hatada dur

# Renkli çıktı için
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Klasör yapısını kontrol et/oluştur
echo -e "${BLUE}Klasör yapısı hazırlanıyor...${NC}"
mkdir -p output/features
mkdir -p output/models
mkdir -p output/reports

# Global değişkenler
PARTIAL_MODE=false
SKIP_MODELS=false

# 1. Feature dosyalarını birleştirme
combine_features() {
    echo -e "\n${BLUE}=== 1. Feature dosyalarını birleştirme =====${NC}"
    echo -e "${YELLOW}Özellikler birleştiriliyor...${NC}"
    
    # output/features klasöründe dosya olup olmadığını kontrol et
    feature_files_count=$(ls -1 output/features/*.csv 2>/dev/null | wc -l)
    
    if [ "$feature_files_count" -eq 0 ]; then
        echo -e "${RED}❌ 'output/features/' klasöründe hiç CSV dosyası bulunamadı!${NC}"
        echo -e "${YELLOW}   Lütfen önce feature extraction adımını çalıştırın.${NC}"
        exit 1
    fi
    
    python scripts/combine_features.py --input-dir "output/features" --output "output/features_combined.csv"
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Feature dosyalarının birleştirilmesinde hata! İşlem durduruluyor.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Feature dosyaları birleştirildi${NC}"
}

# 2. Dengeli veri seti oluşturma
create_balanced_dataset() {
    if [ "$PARTIAL_MODE" = true ]; then
        return
    fi
    
    echo -e "\n${BLUE}=== 2. Dengeli veri seti oluşturma =====${NC}"
    echo -e "${YELLOW}Özellikler dengeleniyor...${NC}"
    
    # combined_features.csv kontrolü
    if [ ! -f "output/features_combined.csv" ]; then
        echo -e "${RED}❌ 'output/features_combined.csv' dosyası bulunamadı!${NC}"
        echo -e "${YELLOW}   Tüm pipeline'ı çalıştırmak yerine her adımı ayrı ayrı çalıştırmayı deneyin.${NC}"
        exit 1
    fi
    
    python scripts/create_balanced_dataset.py --input "output/features_combined.csv" --output "output/balanced_dataset.csv" --ratio 2
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Veri seti dengeleme adımında hata! İşlem durduruluyor.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Dengeli veri seti oluşturuldu${NC}"
}

# 3. Özellik seçimi
select_features() {
    if [ "$PARTIAL_MODE" = true ]; then
        return
    fi
    
    echo -e "\n${BLUE}=== 3. Özellik seçimi =====${NC}"
    echo -e "${YELLOW}Önemli özellikler seçiliyor...${NC}"
    
    # balanced_dataset.csv kontrolü
    if [ ! -f "output/balanced_dataset.csv" ]; then
        echo -e "${RED}❌ 'output/balanced_dataset.csv' dosyası bulunamadı!${NC}"
        exit 1
    fi
    
    python scripts/feature_selection.py --input "output/balanced_dataset.csv" --output "output/selected_features.csv" --method importance --n_features 30 --plot
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Özellik seçimi adımında hata! İşlem durduruluyor.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Özellik seçimi tamamlandı${NC}"
}

# 4. Normalizasyon
normalize_features() {
    if [ "$PARTIAL_MODE" = true ]; then
        return
    fi
    
    echo -e "\n${BLUE}=== 4. Özellik normalizasyonu =====${NC}"
    echo -e "${YELLOW}Özellikler normalize ediliyor...${NC}"
    
    # selected_features.csv kontrolü
    if [ ! -f "output/selected_features.csv" ]; then
        echo -e "${RED}❌ 'output/selected_features.csv' dosyası bulunamadı!${NC}"
        exit 1
    fi
    
    python scripts/normalize_features.py --input "output/selected_features.csv" --output "output/normalized_features.csv" --method standard --plot
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Normalizasyon adımında hata! İşlem durduruluyor.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Normalizasyon tamamlandı${NC}"
}

# 5. NaN Değerlerini Temizleme
clean_data() {
    if [ "$PARTIAL_MODE" = true ]; then
        return
    fi
    
    echo -e "\n${BLUE}=== 5. NaN değerlerini temizleme =====${NC}"
    echo -e "${YELLOW}Eksik değerler temizleniyor...${NC}"
    
    # normalized_features.csv kontrolü
    if [ ! -f "output/normalized_features.csv" ]; then
        echo -e "${RED}❌ 'output/normalized_features.csv' dosyası bulunamadı!${NC}"
        exit 1
    fi
    
    python scripts/clean_data.py --input "output/normalized_features.csv" --output "output/cleaned_features.csv" --method fill_zero --plot
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Veri temizleme adımında hata! İşlem durduruluyor.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Veri temizleme tamamlandı${NC}"
}

# 6. Cross-validation ve model eğitimi
train_models() {
    if [ "$PARTIAL_MODE" = true ]; then
        return
    fi
    
    if [ "$SKIP_MODELS" = true ]; then
        echo -e "\n${YELLOW}Model eğitim adımı atlanıyor (--skip-models)${NC}"
        return
    fi
    
    echo -e "\n${BLUE}=== 6. Model eğitimi ve değerlendirme =====${NC}"
    
    # cleaned_features.csv kontrolü
    if [ ! -f "output/cleaned_features.csv" ]; then
        echo -e "${RED}❌ 'output/cleaned_features.csv' dosyası bulunamadı!${NC}"
        exit 1
    fi
    
    # Farklı model tipleri için cross-validation
    models=("rf" "svm" "knn" "dt")
    
    for model in "${models[@]}"; do
        echo -e "${YELLOW}$model modeli eğitiliyor...${NC}"
        python scripts/cross_validation.py --input "output/cleaned_features.csv" --output-dir "output/models" --model "$model" --k-fold 5 --grid-search --nan-handling fill_zero
        
        if [ $? -ne 0 ]; then
            echo -e "${RED}❌ $model model eğitiminde hata!${NC}"
        else
            echo -e "${GREEN}✓ $model modeli başarıyla eğitildi ve değerlendirildi${NC}"
        fi
    done
}

# 7. JMP formatına dönüştürme
prepare_for_jmp() {
    if [ "$PARTIAL_MODE" = true ]; then
        return
    fi
    
    echo -e "\n${BLUE}=== 7. JMP için veri hazırlama =====${NC}"
    echo -e "${YELLOW}Veri seti JMP için hazırlanıyor...${NC}"
    
    # balanced_dataset.csv kontrolü
    if [ ! -f "output/balanced_dataset.csv" ]; then
        echo -e "${RED}❌ 'output/balanced_dataset.csv' dosyası bulunamadı!${NC}"
        exit 1
    fi
    
    python scripts/format_for_jmp.py --input "output/balanced_dataset.csv" --output "output/balanced_dataset_jmp.csv"
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ JMP hazırlık adımında hata!${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ JMP verisi hazırlandı${NC}"
}

# Yardım mesajını göster
show_help() {
    echo "Kullanım: ./run_from_features.sh [SEÇENEKler]"
    echo ""
    echo "Seçenekler:"
    echo "  --skip-models           Model eğitim adımını atla"
    echo "  --help, -h              Bu yardım mesajını göster"
    echo ""
    echo "Açıklama:"
    echo "  Bu script, output/features/ klasöründeki hazır feature dosyalarını kullanarak analiz pipeline'ını çalıştırır."
    echo "  Önce, tüm feature CSV dosyaları birleştirilir, daha sonra kalan adımlar sırayla çalıştırılır."
    exit 0
}

# Ana fonksiyon
main() {
    echo -e "${BLUE}===================================================${NC}"
    echo -e "${BLUE}       EEG Analiz Boru Hattı Başlatılıyor          ${NC}"
    echo -e "${BLUE}      (Feature Klasöründen Başlayarak)             ${NC}"
    echo -e "${BLUE}===================================================${NC}"
    
    # Başlangıç zamanı
    start_time=$(date +%s)
    
    # İşleme adımlarını çalıştır (feature extraction hariç)
    combine_features
    create_balanced_dataset
    select_features
    normalize_features
    clean_data
    train_models
    prepare_for_jmp
    
    # Bitiş zamanı ve toplam süre
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    minutes=$((duration / 60))
    seconds=$((duration % 60))
    
    if [ "$PARTIAL_MODE" = true ]; then
        echo -e "\n${GREEN}==================================================${NC}"
        echo -e "${GREEN}    ✓ Kısmi işlem başarıyla tamamlandı!          ${NC}"
        echo -e "${GREEN}    Toplam süre: ${minutes} dakika ${seconds} saniye${NC}"
        echo -e "${GREEN}==================================================${NC}"
    else
        echo -e "\n${GREEN}==================================================${NC}"
        echo -e "${GREEN}    ✓ Tüm işlemler başarıyla tamamlandı!         ${NC}"
        echo -e "${GREEN}    Toplam süre: ${minutes} dakika ${seconds} saniye${NC}"
        echo -e "${GREEN}==================================================${NC}"
        echo -e "${YELLOW}Dosya konumları:${NC}"
        echo -e "  - Birleştirilmiş özellikler: ${BLUE}output/features_combined.csv${NC}"
        echo -e "  - Dengeli veri seti: ${BLUE}output/balanced_dataset.csv${NC}"
        echo -e "  - Seçilmiş özellikler: ${BLUE}output/selected_features.csv${NC}"
        echo -e "  - Normalize edilmiş veri: ${BLUE}output/normalized_features.csv${NC}"
        echo -e "  - JMP verisi: ${BLUE}output/balanced_dataset_jmp.csv${NC}"
        echo -e "  - Modeller: ${BLUE}output/models/${NC}"
    fi
}

# Script argümanlarını işle
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-models)
            SKIP_MODELS=true
            shift
            ;;
        --help|-h)
            show_help
            ;;
        *)
            echo -e "${RED}Hata: Bilinmeyen argüman: $1${NC}"
            show_help
            ;;
    esac
done

# Ana fonksiyonu çağır
main 