#!/bin/bash
# run_patient.sh
# Belirli bir hasta için pipeline'ı çalıştırır.

# Kullanım bilgisi
if [ $# -lt 1 ]; then
    echo "Kullanım: ./scripts/run_patient.sh PATIENT_ID [MAX_FILES]"
    echo "  PATIENT_ID: İşlenecek hasta ID (örn: chb01)"
    echo "  MAX_FILES:  İşlenecek maksimum dosya sayısı (isteğe bağlı, varsayılan: tümü)"
    exit 1
fi

PATIENT_ID=$1
MAX_FILES=${2:-9999}  # Varsayılan: tüm dosyalar

# Ana dizine geç
cd "$(dirname "$0")/.."

# Renkli çıktı için
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}===================================================${NC}"
echo -e "${BLUE}  $PATIENT_ID Hasta Verilerini İşleme Başlatılıyor  ${NC}"
echo -e "${BLUE}===================================================${NC}"

# Hasta dizinini kontrol et
PATIENT_DIR="data/physionet.org/files/chbmit/1.0.0/$PATIENT_ID"
if [ ! -d "$PATIENT_DIR" ]; then
    echo -e "${RED}❌ Hata: $PATIENT_ID hasta dizini bulunamadı!${NC}"
    echo -e "   Kontrol edildi: $PATIENT_DIR"
    exit 1
fi

# EDF dosyalarını say
EDF_FILES=($(ls "$PATIENT_DIR"/*.edf 2>/dev/null))
if [ ${#EDF_FILES[@]} -eq 0 ]; then
    echo -e "${RED}❌ Hata: $PATIENT_ID için EDF dosyası bulunamadı!${NC}"
    exit 1
fi

echo -e "${YELLOW}$PATIENT_ID için ${#EDF_FILES[@]} EDF dosyası bulundu.${NC}"
echo -e "${YELLOW}En fazla $MAX_FILES dosya işlenecek.${NC}"

# Çıktı dizinini oluştur
OUTPUT_DIR="output/patient_$PATIENT_ID"
mkdir -p "$OUTPUT_DIR/features"
echo -e "${GREEN}✓ Çıktı dizini oluşturuldu: $OUTPUT_DIR${NC}"

# Hasta verileri için başlangıç zamanı
start_time=$(date +%s)

# Dosya sayacı
file_count=0
processed_count=0
error_count=0

# Çalışma dizini oluştur
WORKING_DIR="$OUTPUT_DIR/working"
mkdir -p "$WORKING_DIR"

# Her EDF dosyasını işle
for edf_file in "${EDF_FILES[@]}"; do
    # Maksimum dosya sayısını kontrol et
    if [ $file_count -ge $MAX_FILES ]; then
        echo -e "${YELLOW}Maksimum dosya sayısına ulaşıldı ($MAX_FILES).${NC}"
        break
    fi
    
    file_count=$((file_count + 1))
    file_name=$(basename "$edf_file")
    
    echo -e "\n${BLUE}[$file_count/${#EDF_FILES[@]}] İşleniyor: $file_name${NC}"
    
    # Özellik çıkarma
    python scripts/extract_all_features.py --file "$edf_file" --output-dir "$OUTPUT_DIR/features"
    
    if [ $? -eq 0 ]; then
        processed_count=$((processed_count + 1))
        echo -e "${GREEN}✓ $file_name başarıyla işlendi${NC}"
    else
        error_count=$((error_count + 1))
        echo -e "${RED}❌ $file_name işlenirken hata oluştu${NC}"
    fi
done

# Tüm özellik dosyalarını birleştir
echo -e "\n${YELLOW}Özellik dosyaları birleştiriliyor...${NC}"
feature_files=($OUTPUT_DIR/features/*.csv)

if [ ${#feature_files[@]} -eq 0 ]; then
    echo -e "${RED}❌ Hiç özellik dosyası bulunamadı!${NC}"
    exit 1
fi

# İlk dosyanın başlıklarını al
head -n 1 "${feature_files[0]}" > "$OUTPUT_DIR/${PATIENT_ID}_features_combined.csv"
# Tüm dosyaların içeriğini (başlık hariç) birleştir
for f in "${feature_files[@]}"; do
    tail -n +2 "$f" >> "$OUTPUT_DIR/${PATIENT_ID}_features_combined.csv"
done

echo -e "${GREEN}✓ Özellik dosyaları birleştirildi: $OUTPUT_DIR/${PATIENT_ID}_features_combined.csv${NC}"

# Dengeli veri seti oluştur
echo -e "\n${YELLOW}Dengeli veri seti oluşturuluyor...${NC}"
python scripts/create_balanced_dataset.py --input "$OUTPUT_DIR/${PATIENT_ID}_features_combined.csv" --output "$OUTPUT_DIR/${PATIENT_ID}_balanced.csv"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Dengeli veri seti oluşturuldu${NC}"
    
    # Özellik seçimi
    echo -e "\n${YELLOW}Özellik seçimi yapılıyor...${NC}"
    python scripts/feature_selection.py --input "$OUTPUT_DIR/${PATIENT_ID}_balanced.csv" --output "$OUTPUT_DIR/${PATIENT_ID}_selected.csv" --method importance --n_features 30 --plot
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Özellik seçimi tamamlandı${NC}"
        
        # JMP formatına dönüştür
        echo -e "\n${YELLOW}JMP formatına dönüştürülüyor...${NC}"
        python scripts/format_for_jmp.py --input "$OUTPUT_DIR/${PATIENT_ID}_balanced.csv" --output "$OUTPUT_DIR/${PATIENT_ID}_jmp.csv"
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ JMP formatına dönüştürüldü${NC}"
        else
            echo -e "${RED}❌ JMP formatına dönüştürme adımında hata oluştu${NC}"
        fi
    else
        echo -e "${RED}❌ Özellik seçimi adımında hata oluştu${NC}"
    fi
else
    echo -e "${RED}❌ Dengeli veri seti oluşturma adımında hata oluştu${NC}"
fi

# İşlem süresi ve özet
end_time=$(date +%s)
duration=$((end_time - start_time))
minutes=$((duration / 60))
seconds=$((duration % 60))

echo -e "\n${GREEN}===================================================${NC}"
echo -e "${GREEN}    ✓ $PATIENT_ID İşlem Özeti                      ${NC}"
echo -e "${GREEN}===================================================${NC}"
echo -e "${YELLOW}Toplam dosya: ${#EDF_FILES[@]}${NC}"
echo -e "${GREEN}Başarıyla işlenen: $processed_count${NC}"
echo -e "${RED}Hata oluşan: $error_count${NC}"
echo -e "${BLUE}İşlem süresi: ${minutes} dakika ${seconds} saniye${NC}"
echo -e "\n${YELLOW}Oluşturulan dosyalar:${NC}"
echo -e "  - Birleştirilmiş özellikler: ${BLUE}$OUTPUT_DIR/${PATIENT_ID}_features_combined.csv${NC}"
echo -e "  - Dengeli veri seti: ${BLUE}$OUTPUT_DIR/${PATIENT_ID}_balanced.csv${NC}"
echo -e "  - Seçilen özellikler: ${BLUE}$OUTPUT_DIR/${PATIENT_ID}_selected.csv${NC}"
echo -e "  - JMP verisi: ${BLUE}$OUTPUT_DIR/${PATIENT_ID}_jmp.csv${NC}"

# Çalışma dizinini temizle
rm -rf "$WORKING_DIR"

exit 0 