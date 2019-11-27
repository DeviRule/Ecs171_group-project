### What is this project about?
This project is about solving the binary classification problem (fraud detection) based on the dataset retrieved from https://www.kaggle.com/mlg-ulb/creditcardfraud. Please check the paper for more details.

### Authors
Kirby Zhou, Wenhao Su, James Lemkin, Zekun Chen, Zhujun Fang,Yuqi Sha, 

Rongfei Li, Hulin Wang, Damu Gao, Bo Xiao, Haoyang Li, Yizhi Huang

### How to run the scripts
The file folder libs consists of multiple utility functions to draw the ROC and PR curves.
Before running the script, it is extremely important to **download corresponding processed Data folder AND dataset from Kaggle website to current workspace.**

Another repository is merged into current repository:  https://github.com/cosmobiosis/fraud_detection

Dataset Folder download link: https://github.com/DeviRule/Ecs171_group-project/tree/master/Data

Kaggle Dataset download link: https://www.kaggle.com/mlg-ulb/creditcardfraud

The output will be automatically saved to current folder or printed out during runtime.

### Outline
The project consists of file scripts, together with utility functions inside the file folder ```libs```.

```RF_new_local.ipynb```: random forest

```decision_tree_grid_search.py```: decision tree

```distribution.py```: preprocessing distribution

```k_nearest_neighbour.ipynb```: KNN

```preprocessing_comparision.ipynb```: SMOTE performance

```Adaboost_main.py```: Adaboost

```SNE_KNN.ipynb```: TSNE and KNN

```ANN_kfold.ipynb```: Feed-Forward Neural Network

### Dependencies
+ **Programming Tools**: *Python3* and *Jupyter Notebook*
+ **Libraries**: *imblearn*, *sklearn*, *Keras*, *matplotlib*, *graphviz* and *pandas*

### Paper and Code connection
In electronic pdf paper file, every link is connected to the corresponding script file which can retrieve and reproduce the results from the paper.

### Support
For any questions contact sensu@ucdavis.edu (email expired in 2021).

### Licence
The project is using MIT license.

### Acknowledgement
This work was supported by course ECS 171 of UC Davis.
The dataset has been collected and analysed during a research collaboration of Worldline and the Machine Learning Group (http://mlg.ulb.ac.be) of ULB (Université Libre de Bruxelles) on big data mining and fraud detection. More details on current and past projects on related topics are available on https://www.researchgate.net/project/Fraud-detection-5 and the page of the DefeatFraud project






