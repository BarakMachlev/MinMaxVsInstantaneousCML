
# TransformerCMLSamplingProtocols
A python toolbox based on PyTorch which utilized neural network for rain estimation and classification from commercial microwave link (CMLs) data. This toolbox provides an implementation of algorithms for extracting rain-rate using neural networks and CMLs.

This project is an extended version of the [PyNNcml](https://github.com/haihabi/PyNNcml) rainfall estimation toolbox, developed as part of the M.Sc. thesis of Hai Victor Habi under the supervision of Prof. Hagit Messer at Tel Aviv University and maintained by the [Cellular Environmental Monitoring and Processing Lab](https://cellenmonlab.tau.ac.il/).  

This version includes new features and experiments as part of the M.Sc. thesis of Barak Machlev under the supervision of Dr. Jonatan Ostrometzky, which focuses on the impact of sampling interval on rainfall estimation from CMLs using Transformers.

![plot](./appendix/cml_figure.png)
* The figure above was create by Jonatan Ostrometzky.


# Getting Started
## Installation
PyNNcml is available on PyPI. To install, run the following command:
```
pip install pynncml
```

## Supported Python Versions



| Python Version | Status                                                                                                |
|----------------|-------------------------------------------------------------------------------------------------------|
| Python 3.9     | ![Run Tests](https://github.com/haihabi/PyNNcml/actions/workflows/run_test_python_3_9.yml/badge.svg)  |
| Python 3.10    | ![Run Tests](https://github.com/haihabi/PyNNcml/actions/workflows/run_test_python_3_10.yml/badge.svg) |
| Python 3.11    | ![Run Tests](https://github.com/haihabi/PyNNcml/actions/workflows/run_test_python_3_11.yml/badge.svg) |
| Python 3.12    | ![Run Tests](https://github.com/haihabi/PyNNcml/actions/workflows/run_test_python_3_12.yml/badge.svg) |


# PyNNcml Features

## Task and Algorithms

1. Wet Dry Classification (RNN[1,2,3] and STD Window[7])
2. Rain Estimation (Constant Baseline [7], Dynamic Baseline [6], Direct RNN Estimation [4,3])
3. Rain Field Interpolation (IDW, GMZ [10])
4. Wet Dry Classification and Rain Rate Estimation [5]


## Datasets
This repository uses the PyTorch version of the OpenMRG dataset [10] and the synthesized datasets generated as part of Barak Machlev’s M.Sc. study, described in detail in the study and used for evaluating the impact of sampling interval on rainfall estimation.

# Contributing

This fork aims to stay up to date and welcomes contributions.
Open a pull request or issue, or email me at barakmachlev@gmail.com.


# References

Please cite one of following paper if you found our neural network model useful. Thanks!

[1] Habi, Hai Victor and Messer, Hagit. "Wet-Dry Classification Using LSTM and Commercial Microwave Links"

[2] Habi, Hai Victor and Messer, Hagit. "RNN MODELS FOR RAIN DETECTION"

[3] Habi, Hai Victor. "Rain Detection and Estimation Using Recurrent Neural Network and Commercial Microwave Links"

[4] Habi, Hai Victor, and Hagit Messer. "Recurrent neural network for rain estimation using commercial microwave links." IEEE Transactions on Geoscience and Remote Sensing 59.5 (2020): 3672-3681.

[5] Barak Machlev, Hai Victor Habi, Hagit Messer, and Jonatan Ostrometzky, “A universal transformer-based algorithm for rain estimation using commercial microwave links across different sampling intervals,” IEEE Geoscience and Remote Sens-ing Letters, 2026.

Also, this package contains the implementations of the following papers:

[6] J. Ostrometzky and H. Messer, “Dynamic determination of the baselinelevel in microwave links for rain monitoring from minimum attenuationvalues,”IEEE Journal of Selected Topics in Applied Earth Observationsand Remote Sensing, vol. 11, no. 1, pp. 24–33, Jan 2018.

[7] M. Schleiss and A. Berne, “Identification of dry and rainy periods using telecommunication  microwave  links,”IEEE  Geoscience  and  RemoteSensing Letters, vol. 7, no. 3, pp. 611–615, 2010

[8] Jonatan Ostrometzky, Adam Eshel, Pinhas Alpert, and Hagit Messer. Induced bias in attenuation measurements taken from commercial microwave links. In 2017 IEEE International
Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 3744–3748. IEEE,2017. <br>

[9] Jonatan Ostrometzky, Roi Raich, Adam Eshel, and Hagit Messer.
Calibration of the
attenuation-rain rate power-law parameters using measurements from commercial microwave networks. In 2016 IEEE International Conference on Acoustics, Speech and Signal
Processing (ICASSP), pages 3736–3740. IEEE, 2016.

[10] Goldshtein, Oren, Hagit Messer, and Artem Zinevich. "Rain rate estimation using measurements from commercial telecommunications links." IEEE Transactions on signal processing 57.4 (2009): 1616-1625.


And include PyTorch implementation of the OpenMRG dataset:

[11] van de Beek, Remco CZ, et al. OpenMRG: Open data from Microwave links, Radar, and Gauges for rainfall quantification in Gothenburg, Sweden. No. EGU23-14295. Copernicus Meetings, 2023.

[12] C. Chwala, A. Overeem, E. Øydvin, L. Petersson W˚ardh, J. Seidel, M. Graf, B. Walraven, E. Covi, H. Habi, M. Fencl et al., “Open-source tools for processing opportunistic rainfall sensor data: An overview of existing tools and the new opensense 535 software packages poligrain, pypwsqc and mergeplg,” 2025.


If you found one of those methods usefully please cite.
