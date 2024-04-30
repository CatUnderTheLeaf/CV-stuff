## Overview of Object Detection algorithms

| Algorithm	| Features | Prediction time / image	| Limitations |
| --------- | --------- | ---------------------- | -------------|
| CNN	| Divides the image into multiple regions and then classifies each region into various classes.	| –	| Needs a lot of regions to predict accurately and hence high computation time.| 
| R-CNN	| Uses selective search to generate regions. Extracts around 2000 regions from each image.	| 40-50 seconds	| High computation time as each region is passed to the CNN separately. Also, it uses three different models for making predictions.| 
| Fast R-CNN	| Each image is passed only once to the CNN and feature maps are extracted. Selective search is used on these maps to generate predictions. Combines all the three models used in R-CNN together.	| 2 seconds	| Selective search is slow and hence computation time is still high.| 
| Faster R-CNN	| Replaces the selective search method with region proposal network (RPN) which makes the algorithm much faster.	| 0.2 seconds	| Object proposal takes time and as there are different systems working one after the other, the performance of systems depends on how the previous system has performed.| 
