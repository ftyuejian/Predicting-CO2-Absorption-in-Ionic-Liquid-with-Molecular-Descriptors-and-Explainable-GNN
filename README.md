# Predicting-CO2-Absorption-in-Ionic-Liquid-with-Molecular-Descriptors-and-Explainable-GNN
Data and code for Predicting $CO_2$ Absorption in Ionic Liquid with Molecular Descriptors and Explainable GNN
![paper pipeline](https://github.com/ftyuejian/Predicting-CO2-Absorption-in-Ionic-Liquid-with-Molecular-Descriptors-and-Explainable-GNN/blob/main/figure/overall.png)

## How to run the code
### reproducing the result for predicting properties with shallow machine learning
* Entering /Shallow_Machine_Learning_for_property_prediction, each jupyter notebook contain a reproducing code for each type of shallow machine learning method mentioned in paper
### reproducing the result for predicting properties with GNN
* Entering /GNN_for_property_prediction
* run `python GIN_Runner.py` to reproduce GIN model result
* run `python GAT_Runner.py` to reproduce GAT model result
* run `python GCN_Runner.py` to reproduce GCN model result
* noted that accuracy on test dataset may vary a little bit each time you run the code due to the random spliting for train and test dataset
### reproducing the result for fragment importance explanation with GNN Explainer
* Entering /Explainer_for_ionic_molecule
* run `explain_whole_dataset.ipynb` to visualizing the fragment importance explanation for the whole dataset 
* run `explain_single_ionic_molecule_pair.ipynb ` to visualizing the explanation for single ionic molecule pair in hotmap form
* run `python fragment_explain.py` to reproduce the fragment importance explanation process for the whole dataset

## About the data
* Due to the reason that the original data has been adapted into different form for different task, we separately clean up a dataset with both smiles dictionary and whole dataset and store those file in Original_Dataset


## Reference

If you find the code useful for your research, please consider citing
```bib
@inproceedings{
  Yue2022predictCO2,
  title={Predicting $CO_2$ Absorption in Ionic Liquid with Molecular Descriptors and Explainable Graph Neural Networks},
  author={Yue Jian and Yuyang Wang and Amir Barati Farimani},
  booktitle={},
  year={2022},
  url={}
}
```

This work is built upon some previous papers which might also interest you:

* Song, Yang, and Stefano Ermon. "Generative Modeling by Estimating Gradients of the Data Distribution." *Proceedings of the 33rd Annual Conference on Neural Information Processing Systems*. 2019.
* Song, Yang, and Stefano Ermon. "Improved techniques for training score-based generative models." *Proceedings of the 34th Annual Conference on Neural Information Processing Systems*. 2020.
* Ho, Jonathan, Ajay Jain, and Pieter Abbeel. "Denoising diffusion probabilistic models." *Proceedings of the 34th Annual Conference on Neural Information Processing Systems*. 2020.


