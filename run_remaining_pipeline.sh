#!/bin/bash
echo "1. Features klasöründeki dosyaları birleştirme"
FIRST_FILE=$(ls output/features/*.csv | head -1)
head -1 $FIRST_FILE > output/features_combined.csv
for f in output/features/*.csv; do
  echo "  İşleniyor: $(basename $f)"
  tail -n +2 "$f" >> output/features_combined.csv
done
echo "2. Dengeli veri seti oluşturma"
python scripts/create_balanced_dataset.py --input "output/features_combined.csv" --output "output/balanced_dataset.csv" --ratio 2
echo "3. Özellik seçimi"
python scripts/feature_selection.py --input "output/balanced_dataset.csv" --output "output/selected_features.csv" --method importance --n_features 30 --plot
echo "4. Normalizasyon"
python scripts/normalize_features.py --input "output/selected_features.csv" --output "output/normalized_features.csv" --method standard --plot
echo "5. NaN değerlerini temizleme"
python scripts/clean_data.py --input "output/normalized_features.csv" --output "output/cleaned_features.csv" --method fill_zero --plot
echo "6. Model eğitimi ve değerlendirme"
models=("rf" "svm" "knn" "dt")
for model in "${models[@]}"; do
  echo "  $model modeli eğitiliyor..."
  python scripts/cross_validation.py --input "output/cleaned_features.csv" --output-dir "output/models" --model "$model" --k-fold 5 --grid-search --nan-handling fill_zero
done
echo "7. JMP için veri hazırlama"
python scripts/format_for_jmp.py --input "output/balanced_dataset.csv" --output "output/balanced_dataset_jmp.csv"
echo "İşlem tamamlandı!"
