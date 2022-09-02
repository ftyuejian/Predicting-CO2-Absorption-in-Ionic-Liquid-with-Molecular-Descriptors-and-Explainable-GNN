# Predicting-CO2-Absorption-in-Ionic-Liquid-with-Molecular-Descriptors-and-Explainable-GNN
Data and code for Predicting CO2 Absorption in Ionic Liquid with Molecular Descriptors and Explainable GNN
![paper pipeline](https://github.com/ftyuejian/Predicting-CO2-Absorption-in-Ionic-Liquid-with-Molecular-Descriptors-and-Explainable-GNN/blob/main/figure/overall.png)

## How to run the code
### reproducing the result for predicting properties with shallow machine learning
* Entering /Shallow_Machine_Learning_for_property_prediction, each jupyter notebook contain a reproducing code for each type of shallow machine learning method mentioned in paper
### reproducing the result for predicting properties with GNN
* Entering /GNN_for_property_prediction
* run `python GIN_Runner.py` to reproduce GIN model result
* run `python GAT_Runner.py` to reproduce GAT model result
* run `python GCN_Runner.py` to reproduce GCN model result
### reproducing the result for fragment importance explanation with GNN Explainer


## About the data






## Reference

If you find the code useful for your research, please consider citing
```bib
@inproceedings{
  song2021scorebased,
  title={Score-Based Generative Modeling through Stochastic Differential Equations},
  author={Yang Song and Jascha Sohl-Dickstein and Diederik P Kingma and Abhishek Kumar and Stefano Ermon and Ben Poole},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={https://openreview.net/forum?id=PxTIG12RRHS}
}
```

This work is built upon some previous papers which might also interest you:

* Song, Yang, and Stefano Ermon. "Generative Modeling by Estimating Gradients of the Data Distribution." *Proceedings of the 33rd Annual Conference on Neural Information Processing Systems*. 2019.
* Song, Yang, and Stefano Ermon. "Improved techniques for training score-based generative models." *Proceedings of the 34th Annual Conference on Neural Information Processing Systems*. 2020.
* Ho, Jonathan, Ajay Jain, and Pieter Abbeel. "Denoising diffusion probabilistic models." *Proceedings of the 34th Annual Conference on Neural Information Processing Systems*. 2020.


