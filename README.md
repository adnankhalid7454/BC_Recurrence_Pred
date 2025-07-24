# Breast Cancer Recurrence Prediction using MRI

This repository provides a structured implementation of the dual-stream feature extraction framework that combines **tumor-specific radiomic features**, obtained via automated lesion segmentation, with **global parenchymal descriptors** extracted from the whole breast volume. By combining these imaging features with curated clinicopathological variables, we construct a unified, multimodal representation of recurrence risk.

## Dataset

This study utilizes the publicly available dataset:
- **Dynamic contrast-enhanced MRI of Breast Cancer Patients with Tumor Locations (Duke-Breast-Cancer-MRI)**
- Accessible at: [The Cancer Imaging Archive (TCIA)](https://doi.org/10.7937/TCIA.e3sv-re93)

## Repository Structure
```
./
├── Trained_models/
├── Breast_Segmentation/
│   └── (from Lidia Garrucho's GitHub)
├── Tumor_Segmentation/
│   └── (from Maciej Mazurowski's GitHub)
├── Feature_Extraction_Classification/
│   ├── feature_extraction.py
│   ├── classification_model.py
│   └── requirements.txt
└── README.md
```
## Running the Framework

### Step 1: Perform Breast Segmentation
Run the tumor segmentation as per [Maciej Mazurowski's GitHub instructions](https://github.com/MaciejMazurowski/mri-breast-tumor-segmentation).

### Step 2: Perform Tumor Segmentation
Run the breast segmentation following [Lidia Garrucho's GitHub instructions](https://github.com/LidiaGarrucho/MAMA-MIA).

### Step 3: Feature Extraction and Classification
After the segmentation of the tumor and breast region, feature extraction is performed using the file feature_extraction.py. This script extracts radiomic features from the segmented regions to create a comprehensive dataset for classification purposes.

To test the classification model, these extracted features are used by the script classification_model.py. This script loads a trained model, provided in the same folder, to classify the risk of recurrence based on the extracted features.
```bash
# Run feature extraction
python feature_extraction.py

# Run classification model
python classification_model.py
```

## References

- **Dataset**:
```
@article{saha2018machine,
  title={A machine learning approach to radiogenomics of breast cancer: a study of 922 subjects and 529 DCE-MRI features},
  author={Saha, Ashirbani and Harowicz, Michael R and Grimm, Lars J and Kim, Connie E and Ghate, Sujata V and Walsh, Ruth and Mazurowski, Maciej A},
  journal={British journal of cancer},
  volume={119},
  number={4},
  pages={508--516},
  year={2018},
  publisher={Nature Publishing Group UK London}
}
```
- **Tumor Segmentation**:
```
@misc{garrucho2021mamamia,
  author = {Lidia Garrucho},
  title = {MAMA-MIA: Breast segmentation in MRI},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {https://github.com/LidiaGarrucho/MAMA-MIA},
}
```

## Citation
If you use this work, please cite the following:
```
[paper's citation here once published.]
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
