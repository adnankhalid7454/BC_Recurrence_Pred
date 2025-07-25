## Summary

1. **Feature Extraction**
   - Run `Feature_extraction.ipynb` to extract imaging-based features.
   - Merge them with clinical data for enriched input representation.
2. **Preprocessing**
   - Handle missing values using `knn_imputer.pkl`.
   - Select top features using `feature_selector_kbest.pkl`.
   - Normalize features using `scaler_minmax.pkl`.
3. **Prediction**
   - Use `test_classifier.py` to classify recurrence using the pre-trained TabNet model stored in the `model/` folder.
---

| File                          | Description                                                  |
|-------------------------------|--------------------------------------------------------------|
| `Feature_extraction.ipynb`    | Extracts and merges features from raw and clinical data      |
| `feature_selector_kbest.pkl`  | KBest model for feature selection                            |
| `scaler_minmax.pkl`           | Min-Max scaler for normalization                             |
| `knn_imputer.pkl`             | KNN-based imputer for missing values                         |
| `test_classifier.py`          | Loads model and evaluates recurrence prediction              |
  
jupyter notebook Feature_extraction.ipynb

python test_classifier.py

