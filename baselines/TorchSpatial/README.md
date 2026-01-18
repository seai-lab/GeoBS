# GeoBS: A Python Toolbox for Evaluating and Harnessing Geo-Bias

## Overview
**GeoBS** stands for **Geo**-**B**ias **S**cores. It provides users with a collection of information-theoretic, model-agnostic metrics of geospatial bias (i.e., non-uniform model performance across different regions on Earth). It is complementary to existing model performance metrics such as accuracy, precision, recall and reciprocal rank. 

This project aims at providing a plug-and-play toolbox for researchers to painlessly benchmarking the geo-bias of their models, encouraging them to report the geo-bias scores alongside with other metrics, and using geo-bias scores as training objectives to help harness the geo-bias of models. We hope this will boost the fairness and trustworthiness of spatial data analysis and GeoAI research -- in fact, we find that while introducing fine-grained geospatial information into models greatly improves the performance, it also significantly increases geo-bias. It is important to pay attention to the both sides of the coin. 

Different geo-bias scores focus on different spatial aspects of geospatial bias. In this initial version, we introduce two most important categories of GBS: **Spatial Self-Information (SSI)** Scores and **Spatial Relative Information (SRI)** Scores. As the names indicate, SSI focuses on the bias originating from the spatial arrangement within each region of interest (i.e., _self_-information), while SRI focuses on the bias originating from the performance mis-alignment between different regions of interest (i.e., _relative_ information). Each category of GBS consists of different GBS implementations, further specifying what spatial arrangement/performance mis-alignment we exactly take into consideration. Select the GBS implementation that best fits your research questions.

We will be actively supporting more GBS categories in the future.

## Supported Geo-Bias Scores

![Illustration of Supported Geo-Bias Scores](readme/GBS_5.png)

### SSI Scores
In classic geostatistics, an _unmarked_ measurement only cares about the spatial distribution of data (i.e., the data points are not "marked" with specific values), while a _marked_ measurement additionally cares about the values of data. Following this naming tradition, we have two types of SSI Scores:
#### Unmarked SSI
It only accounts for the spatial arrangement of the observed data.
#### Marked SSI
It comprehensively accounts for both the spatial arrangement of the observed data, and how the high/low performance values scatter through the data. Beware that the Marked SSI Score can not be directly compared with the Unmarked SSI Score.


### SRI Scores
SRI measures the heterogeneity of model performance within a given region of interest (ROI). If a model is not geo-biased, its performance should be similar across the entire ROI and within any local patch of the ROI. Based on how we partition the ROI, we provide 3 types of SRI Scores, each score corresponding to a certain type of spatial heterogeneity.
#### Scale-Grid SRI
Partition the ROI into smaller grids.
#### Distance-Lag SRI
Partition the ROI into concentric distance lags.
#### Direction-Sector SRI
Partition the ROI into sectors.

## Interpreting Geo-Bias Scores

## Harnessing Geo-Bias
