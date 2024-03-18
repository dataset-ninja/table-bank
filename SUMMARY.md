**Table Bank Dataset** is a dataset for instance segmentation and object detection tasks. It is used in the optical character recognition (OCR) domain. 

The dataset consists of 68017 images with 169408 labeled objects belonging to 1 single class (*table*).

Images in the Table Bank dataset have pixel-level instance segmentation and bounding box annotations. Due to the nature of the instance segmentation task, it can be automatically transformed into a semantic segmentation task (only one mask for every class). All images are labeled (i.e. with annotations). There are 3 splits in the dataset: *train* (63633 images), *val* (2470 images), and *test* (1914 images). Alternatively, the dataset could be split into 2 documents: ***word*** (59700 images) and ***latex*** (8317 images). The dataset was released in 2019 by the <span style="font-weight: 600; color: grey; border-bottom: 1px dashed #d3d3d3;">Beihang University, China</span> and <span style="font-weight: 600; color: grey; border-bottom: 1px dashed #d3d3d3;">Microsoft Research Asia, China</span>.

<img src="https://github.com/dataset-ninja/table-bank/raw/main/visualizations/poster.png">
