# Indian Driving Dataset-Detections (IDD-D): YOLOv11 Format.

[![License](https://img.shields.io/badge/License-Research%20Use%20Only-lightgrey.svg)](https://idd.insaan.iiit.ac.in/dataset/details/)
[![YOLOv11 Format](https://img.shields.io/badge/Format-YOLOv11-%2344aa11)](https://ultralytics.com)
[![Paper](https://img.shields.io/badge/paper-arXiv-B31B1B.svg)](https://arxiv.org/abs/1811.10244)

This repository contains the **[Indian Driving Dataset (IDD)](https://idd.insaan.iiit.ac.in/)**, restructured and converted into the YOLOv11 format for object detection. This version was created to facilitate training and benchmarking object detection models on road scenes from India, which feature unique vehicle types and challenging traffic conditions.

---

## Dataset Overview:
- **Purpose**: To provide a high quality, large scale dataset for training and evaluating object detection models (like YOLOv11) specifically on Indian roads.
- **Key Features**:
  - **Diverse Classes**: Includes object classes common in India, such as `autorickshaw`, `rider`, and `animal`.
  - **Challenging Scenarios**: Captures a wide variety of scenes, including varying weather, lighting conditions, and heavy traffic.
  - **Cleaned and Restructured**: The dataset has been cleaned and organized for seamless integration with modern YOLO based training pipelines.
  - **Standardized Splits**: The data is split into training, validation, and test sets following an 8:1:1 ratio.

---

## Dataset Details:
### Source:
This dataset is a converted version of the original **Indian Driving Dataset-Detections (IDD-D)**. All credits for the original data collection and annotation go to the creators at IIIT Hyderabad.

- **Homepage**: [https://idd.insaan.iiit.ac.in/](https://idd.insaan.iiit.ac.in/)
- **Original Paper**: [IDD: A Dataset for Exploring Problems of Autonomous Navigation in Unconstrained Environments](https://arxiv.org/abs/1811.10244)

### Authors
- **Original Authors**: Girish Varma, Anbumani Subramanian, Namburi Tejaswi, Anoop M. Namboodiri, Manuj M. Sharma, C.V. Jawahar.
- **YOLOv11 Conversion and Restructuring**: Mridankan Mandal.

### Modifications by Mridankan Mandal:
- **YOLOv11 Conversion**: All annotations were converted from the original format to YOLO `.txt` files with normalized bounding box coordinates (`class x_center y_center width height`).
- **Restructuring**: The directory structure was reorganized to be compatible with standard YOLO training scripts.
- **Data Cleaning**: The dataset was reviewed to ensure annotation consistency and integrity for object detection tasks.
- **Test Set Handling**: The original unlabeled test set from IDD was excluded. A new, labeled test set was created from the original training data to enable comprehensive model evaluation and self benchmarking.

### Classes:
The dataset contains 15 object classes relevant to Indian driving scenarios.

| Class ID | Class Name       |
| -------- | ---------------- |
| 0        | animal           |
| 1        | autorickshaw     |
| 2        | bicycle          |
| 3        | bus              |
| 4        | car              |
| 5        | caravan          |
| 6        | motorcycle       |
| 7        | person           |
| 8        | rider            |
| 9        | traffic light    |
| 10       | traffic sign     |
| 11       | trailer          |
| 12       | train            |
| 13       | truck            |
| 14       | vehicle fallback |

**Total Classes**: 15

### Statistics
- **Total Images**: 41,962
  - **Train**: 33,569 images
  - **Validation**: 4,196 images
  - **Test**: 4,197 images
- **Split Ratio**: Approximately 80% (Train) / 10% (Validation) / 10% (Test)

---

## Dataset Structure

    IDDDetectionsYOLODataset/
    ├── train/
    │   ├── images/         #33569 images
    │   └── labels/         #33569 labels
    ├── val/
    │   ├── images/         #4196 images
    │   └── labels/         #4196 labels
    ├── test/
    │   ├── images/         #4197 images
    │   └── labels/         #4197 labels
    ├── data.yaml           #The YOLO dataset configuration file.
    ├── license.md          #The license file.
    └── ReadMe.md           #This documentation file.

### Explanation:
- **`train/`, `val/` and `test/`**: Contain the split datasets for model training and evaluation.
- **`images/`**: Contains the input `.png` or `.jpg` images.
- **`labels/`**: Contains YOLO format `.txt` files with corresponding bounding box annotations.
- **`data.yaml`**: The configuration file that defines dataset paths, number of classes, and class names for YOLO training.

---

### Example `data.yaml`:
```yaml
#Indian Driving Dataset-Detections (IDD-D) - YOLOv11 Configuration
path: ../IDDDetectionsYOLODataset/  #Path to dataset root from YOLOv11 project
train: train/images
val: val/images
test: test/images

#Class information
nc: 15 #number of classes
names:
- animal
- autorickshaw
- bicycle
- bus
- car
- caravan
- motorcycle
- person
- rider
- traffic light
- traffic sign
- trailer
- train
- truck
- vehicle fallback
````

-----

## License

The original IDD dataset is released for **academic research purposes only**. Any derivative works, including this YOLO formatted version, are subject to the same non-commercial restriction. Please refer to the official IDD website for complete license details.

-----

## Citation

If you use this dataset in your research, please cite the original IDD paper:

```bibtex
@inproceedings{Varma_2019_WACV,
    author = {Varma, Girish and Subramanian, Anbumani and Namburi, Tejaswi and Namboodiri, Anoop M. and Sharma, Manuj M. and Jawahar, C.V.},
    title = {{IDD: A Dataset for Exploring Problems of Autonomous Navigation in Unconstrained Environments}},
    booktitle = {The IEEE Winter Conference on Applications of Computer Vision (WACV)},
    month = {January},
    year = {2019}
}
```

-----

## Acknowledgements

We extend our sincere gratitude to the creators of the Indian Driving Dataset at IIIT Hyderabad for their invaluable contribution to the computer vision community.
