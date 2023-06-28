# **HeatMapper:** Modeling urban heat intensity

<p align="center">
    <img src="figures/lst_munich.JPG" alt="drawing"  width="1000"/>
    <br>
    &copy; OpenStreetMap
</p>

## Background

Increasing urbanisation and climate change have led to the evolving and intensification of urban heat islands which can be defined as local areas where the temperature is considerably higher than in their vicinity. High temperatures in general are inevitably associated with partly drastical consequences for human health. However, heat reduction measures are available that can deal with the urban heat island effect: Increasing vegetation, cutting the amount of impervious surfaces, etc.. The goal of this project is to identify heat islands by analysing applicable data for the city of Munich and to model the impact of additional heat reduction measures on potential temperature occurences.

## Approach

This project uses land surface temperature data from Ecostress and official property data as well as orthophotos from the Bavarian State Office for Digitisation, Broadband and Surveying. The former data source denotes the dependent variable in our analysis. The latter two data sources were used to extract land cover / land usage (LCLU) characteristics forming the basis of our feature set. We used pre-trained and fine-tuned neural networks to reach a granular segregation of land cover to also detect patterns that are not stored in official data.

<p align="center">
    <img src="figures/grid_element_all.JPG" alt="drawing"  width="800"/>
    <br>
    &copy; OpenStreetMap
</p>

A linear regression was then used to model the relationship between a neighbourhood's temperature and its surface.

More details on the scientific background including academic references that we followed along can be found [here](https://ds-project-modeling-urban-heat-intensity.tiiny.site/).

### App

We provide an [interative application](https://github.com/MGenschow/DS_Project) that identifies urban heat islands in the city of Munich and also demonstrates the impact of potential heat mitigating effects by hypothetically changing the surfaces of an urban area. In addition, more extensive information about our project is presented there.

### Code 
We have only used the programming language Python in the context of this research project. This Github repository hosts all the code from data acquisition to statistical modeling to app source code. 

<br>

---
This project was part of the module DS500 "Data Science Project" of the Data Science in Business and Economics Master degree at the University of Tuebingen.
