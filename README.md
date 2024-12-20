# Airbnb NYC 2019 Analysis

This repository contains an in-depth analysis of the Airbnb NYC 2019 dataset. The project covers various statistical and machine learning techniques used to understand pricing trends, room types, neighborhood group variations, and other factors affecting Airbnb listings in New York City.

## Features

- **Exploratory Data Analysis (EDA)**: Initial analysis of the dataset, including data summary, missing values, and basic visualizations.
- **Statistical Hypothesis Testing**:
  - T-tests to compare prices across neighborhoods.
  - ANOVA to assess the price variation based on room types.
  - Correlation analysis between price and other variables.
- **Machine Learning Models**:
  - **Linear Regression**: Used to predict Airbnb prices based on various factors.
  - **Random Forest**: A robust machine learning model to predict Airbnb prices.
- **Data Visualizations**: Includes various plots created using ggplot2 to visualize distributions, relationships, and trends.

## Dataset

The dataset used in this project is the **Airbnb NYC 2019** dataset, which includes listings from Airbnb in New York City. The dataset features the following columns:

- **price**: The price of the listing.
- **neighbourhood_group**: The neighborhood group the listing belongs to.
- **room_type**: The type of room (e.g., Entire home/apt, Private room, Shared room).
- **latitude**: Latitude of the listing.
- **longitude**: Longitude of the listing.
- **reviews_per_month**: The average number of reviews per month.
- **last_review**: The last date a review was made.

## Getting Started

### Prerequisites

To run the code, you need to install the following R packages:

```r
install.packages("dplyr")
install.packages("ggplot2")
install.packages("skimr")
install.packages("ggthemes")
install.packages("tidyverse")
install.packages("caret")
install.packages("randomForest")


