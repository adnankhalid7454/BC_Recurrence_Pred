# Dual-Stream Breast MRI Feature Extraction and Classification

This repository provides a structured implementation of the dual-stream feature extraction framework introduced in our research paper. The framework combines **tumor-specific radiomic features**, obtained via automated lesion segmentation, with **global parenchymal descriptors** extracted from the whole breast volume. By combining these imaging features with curated clinicopathological variables, we construct a unified, multimodal representation of recurrence risk.

## Dataset

This study utilizes the publicly available dataset:
- **Dynamic contrast-enhanced MRI of Breast Cancer Patients with Tumor Locations (Duke-Breast-Cancer-MRI)**
- Accessible at: [The Cancer Imaging Archive (TCIA)](https://doi.org/10.7937/TCIA.e3sv-re93)

## Repository Structure
```
./
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
Saha, Ashirbani, Grimm, Lars J., Ghate, Sujata V., Kim, Connie E., Soo, Mary Scott, Yoon, Seung Jong, & Mazurowski, Maciej A. (2021). Dynamic contrast-enhanced magnetic resonance images of breast cancer patients with tumor locations [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/TCIA.e3sv-re93
```
- **Breast Segmentation**:
```
@article{saha2018machine,
  title={Machine learning-based prediction of future breast cancer using algorithmically measured background parenchymal enhancement on high-risk screening MRI},
  author={Saha, Ashirbani and Grimm, Lars J and Ghate, Sujata V and Kim, Connie E and Soo, Mary Scott and Yoon, Seung Jong and Mazurowski, Maciej A},
  journal={Journal of Magnetic Resonance Imaging},
  volume={47},
  number={1},
  pages={223--231},
  year={2018},
  publisher={Wiley Online Library}
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

- **Your Paper**:
```If you use this work, please cite the following:
[paper's citation here once published.]
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
