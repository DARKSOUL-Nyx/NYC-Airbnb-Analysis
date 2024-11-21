# ===============================================================
# Airbnb NYC 2019 Analysis: Exploratory Data Analysis & Modeling
# Author: [VSS NISHWAN]
# GitHub: [https://github.com/DARKSOUL-Nyx/NYC-Airbnb-Analysis]
# Description: This script provides data exploration, visualization, 
#              and predictive modeling for Airbnb listings in NYC.
# ===============================================================

# =======================
# 1. Load Required Libraries
# =======================
# Install necessary libraries (only need to run once)
install.packages(c("dplyr", "ggplot2", "skimr", "ggthemes", "tidyverse", "caret", "randomForest"))

# Load libraries
library(dplyr)
library(ggplot2)
library(skimr)
library(ggthemes)
library(tidyverse)
library(caret)
library(randomForest)

# =======================
# 2. Load and Explore Data
# =======================
# Load the dataset
AB_NYC_2019 <- read.csv("AB_NYC_2019.csv")

# Preview structure and summary
glimpse(AB_NYC_2019)
summary(AB_NYC_2019)

# Quick summary using skimr
skim(AB_NYC_2019)

# =======================
# 3. Data Cleaning and Preprocessing
# =======================
# Check for missing values
missing_summary <- colSums(is.na(AB_NYC_2019))
print(missing_summary)

# Handle missing values: Fill NAs in reviews_per_month and convert last_review to Date
AB_NYC_2019 <- AB_NYC_2019 %>%
  mutate(
    reviews_per_month = ifelse(is.na(reviews_per_month), 0, reviews_per_month),
    last_review = as.Date(last_review, format = "%Y-%m-%d")
  )

# Verify missing values are handled
print(colSums(is.na(AB_NYC_2019)))

# =======================
# 4. Exploratory Data Analysis (EDA)
# =======================

# 4.1 Distribution of Prices
ggplot(AB_NYC_2019, aes(x = price)) +
  geom_histogram(binwidth = 50, fill = "steelblue", color = "black") +
  scale_x_continuous(limits = c(0, 1000)) +
  theme_minimal() +
  labs(title = "Distribution of Airbnb Prices in NYC", x = "Price", y = "Frequency")

# 4.2 Room Type Count
ggplot(AB_NYC_2019, aes(x = room_type, fill = room_type)) +
  geom_bar() +
  theme_minimal() +
  labs(title = "Room Types Available", x = "Room Type", y = "Count")

# 4.3 Average Price by Room Type
AB_NYC_2019 %>%
  group_by(room_type) %>%
  summarise(avg_price = mean(price, na.rm = TRUE)) %>%
  ggplot(aes(x = room_type, y = avg_price, fill = room_type)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "Average Price by Room Type", x = "Room Type", y = "Average Price")

# 4.4 Average Price by Neighborhood Group
AB_NYC_2019 %>%
  group_by(neighbourhood_group) %>%
  summarise(avg_price = mean(price, na.rm = TRUE)) %>%
  ggplot(aes(x = neighbourhood_group, y = avg_price, fill = neighbourhood_group)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "Average Price by Neighborhood Group", x = "Neighborhood Group", y = "Average Price")

# 4.5 Airbnb Listings by Location
ggplot(AB_NYC_2019, aes(x = longitude, y = latitude, color = neighbourhood_group)) +
  geom_point(alpha = 0.5) +
  theme_minimal() +
  labs(title = "Airbnb Listings Across NYC", x = "Longitude", y = "Latitude")

# =======================
# 5. Statistical Tests
# =======================

# 5.1 T-Test: Compare Prices Between Manhattan and Brooklyn
manhattan_prices <- subset(AB_NYC_2019, neighbourhood_group == "Manhattan")$price
brooklyn_prices <- subset(AB_NYC_2019, neighbourhood_group == "Brooklyn")$price

t_test_result <- t.test(manhattan_prices, brooklyn_prices)
print(t_test_result)

if (t_test_result$p.value < 0.05) {
  cat("Significant difference in average prices between Manhattan and Brooklyn.\n")
} else {
  cat("No significant difference in average prices between Manhattan and Brooklyn.\n")
}

# 5.2 ANOVA: Compare Prices Across Room Types
anova_result <- aov(price ~ room_type, data = AB_NYC_2019)
anova_summary <- summary(anova_result)
print(anova_summary)

# =======================
# 6. Predictive Modeling
# =======================

# 6.1 Data Preparation for Modeling
AB_NYC_2019$log_price <- log(AB_NYC_2019$price + 1)
features <- c("log_price", "neighbourhood_group", "room_type", 
              "minimum_nights", "number_of_reviews", 
              "calculated_host_listings_count", "availability_365")
data <- AB_NYC_2019 %>% select(all_of(features))
data$neighbourhood_group <- as.factor(data$neighbourhood_group)
data$room_type <- as.factor(data$room_type)

# Split into Train/Test
set.seed(123)
train_index <- createDataPartition(data$log_price, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# 6.2 Linear Regression Model
lm_model <- lm(log_price ~ ., data = train_data)
lm_predictions <- predict(lm_model, test_data)

# 6.3 Random Forest Model
rf_model <- randomForest(log_price ~ ., data = train_data, ntree = 100)
rf_predictions <- predict(rf_model, test_data)

# 6.4 Model Evaluation
test_data$actual_price <- exp(test_data$log_price) - 1
lm_price_predictions <- exp(lm_predictions) - 1
rf_price_predictions <- exp(rf_predictions) - 1

lm_rmse <- sqrt(mean((test_data$actual_price - lm_price_predictions)^2))
rf_rmse <- sqrt(mean((test_data$actual_price - rf_price_predictions)^2))

lm_r2 <- 1 - sum((test_data$actual_price - lm_price_predictions)^2) /
  sum((test_data$actual_price - mean(test_data$actual_price))^2)
rf_r2 <- 1 - sum((test_data$actual_price - rf_price_predictions)^2) /
  sum((test_data$actual_price - mean(test_data$actual_price))^2)

cat("Linear Regression - RMSE:", lm_rmse, "R-squared:", lm_r2, "\n")
cat("Random Forest - RMSE:", rf_rmse, "R-squared:", rf_r2, "\n")

# =======================
# 7. Clustering Analysis
# =======================
numeric_data <- AB_NYC_2019 %>% select(latitude, longitude, price)
numeric_data <- scale(numeric_data)

set.seed(123)
clusters <- kmeans(numeric_data, centers = 5)
AB_NYC_2019$cluster <- as.factor(clusters$cluster)

ggplot(AB_NYC_2019, aes(x = longitude, y = latitude, color = cluster)) +
  geom_point(alpha = 0.5) +
  theme_minimal() +
  labs(title = "Clustering of Airbnb Listings by Location and Price")

# =======================
# 8. Save Processed Data
# =======================
write.csv(AB_NYC_2019, "Processed_AB_NYC_2019.csv", row.names = FALSE)
