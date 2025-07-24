##  Breast Cancer Recurrence Prediction
Dual-Stream Breast MRI Feature Extraction and Classification
This repository provides a structured implementation of the dual-stream feature extraction framework introduced in our research paper. The framework combines tumor-specific radiomic features, obtained via automated lesion segmentation, with global parenchymal descriptors extracted from the whole breast volume.

Dataset

This study utilizes the publicly available dataset:

Dynamic contrast-enhanced MRI of Breast Cancer Patients with Tumor Locations (Duke-Breast-Cancer-MRI)

Accessible at: The Cancer Imaging Archive (TCIA): https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/

Repository Structure
./
├── Tumor_Segmentation/
│   └── (from Maciej Mazurowski's GitHub)
├── Breast_Segmentation/
│   └── (from Lidia Garrucho's GitHub)
├── Feature_Extraction_Classification/
│   ├── feature_extraction.py
│   ├── classification_model.py
│   └── requirements.txt
└── README.md
Installation and Setup
1. Clone this repository
git clone <your_repo_url>
cd <your_repo_name>
2. Tumor Segmentation Setup
Clone Maciej Mazurowski's tumor segmentation repository:

cd Tumor_Segmentation
git clone https://github.com/MaciejMazurowski/mri-breast-tumor-segmentation.git
cd mri-breast-tumor-segmentation
pip install -r requirements.txt
3. Breast Segmentation Setup
Clone Lidia Garrucho's breast segmentation repository:

cd ../../Breast_Segmentation
git clone https://github.com/LidiaGarrucho/MAMA-MIA.git
cd MAMA-MIA
pip install -r requirements.txt
4. Feature Extraction and Classification Setup
Navigate to the dedicated folder and install requirements:

cd ../../Feature_Extraction_Classification
pip install -r requirements.txt
Ensure your directory paths in scripts are correctly adjusted to your dataset and segmentation outputs.

Running the Framework
Step 1: Perform Tumor Segmentation
Run tumor segmentation as per Maciej Mazurowski's GitHub instructions.

Step 2: Perform Breast Segmentation
Run breast segmentation following Lidia Garrucho's GitHub instructions.

Step 3: Feature Extraction and Classification
cd ../Feature_Extraction_Classification

# Run feature extraction
python feature_extraction.py

# Run classification model
python classification_model.py

# References
Dataset:
Saha, Ashirbani, Grimm, Lars J., Ghate, Sujata V., Kim, Connie E., Soo, Mary Scott, Yoon, Seung Jong, & Mazurowski, Maciej A. (2021). Dynamic contrast-enhanced magnetic resonance images of breast cancer patients with tumor locations [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/TCIA.e3sv-re93
Tumor Segmentation:
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
Breast Segmentation:
@misc{garrucho2021mamamia,
  author = {Lidia Garrucho},
  title = {MAMA-MIA: Breast segmentation in MRI},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {https://github.com/LidiaGarrucho/MAMA-MIA},
}
# Citation
If you use this work, please cite the following:
Your Paper:
License
This project is licensed under the MIT License - see the LICENSE file for details.
