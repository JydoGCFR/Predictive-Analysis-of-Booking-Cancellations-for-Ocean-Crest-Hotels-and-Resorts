# Required libraries
library(cluster)  # For clustering algorithms
library(factoextra)  # For visualization of clusters
library(dendextend)  # For advanced dendrogram visualization
library(dplyr)

# Step 1: Data Preparation
# Read the data (adjust file path as needed)
hotel_data <- read.csv("OceanCrestModified.csv", header = TRUE)

# Select key features for clustering
features_for_clustering <- hotel_data %>%
  select(lead_time, adr, total_of_special_requests, deposit_type, customer_type, market_segment)

# Step 2: Convert categorical variables into numeric values
features_for_clustering$deposit_type <- as.numeric(as.factor(features_for_clustering$deposit_type))
features_for_clustering$customer_type <- as.numeric(as.factor(features_for_clustering$customer_type))
features_for_clustering$market_segment <- as.numeric(as.factor(features_for_clustering$market_segment))

# Step 3: Sampling Data
# Randomly sample 10,000 observations from the data
set.seed(123)  # For reproducibility
sample_size <- 10000  # Define your sample size
hotel_sample <- features_for_clustering[sample(1:nrow(features_for_clustering), sample_size), ]

# Step 4: Scaling the data
# Scale the sampled data to ensure all features have equal weight in the clustering
scaled_sample <- scale(hotel_sample)

# Step 5: Elbow Method and Silhouette Method for Determining Optimal Number of Clusters

# Elbow Method: Determine the optimal number of clusters using the "within-cluster sum of squares" (WSS)
fviz_nbclust(scaled_sample, FUN = hcut, method = "wss")  # wss: within-cluster sum of squares

# Silhouette Method: Determine the optimal number of clusters using the silhouette method
fviz_nbclust(scaled_sample, FUN = hcut, method = "silhouette")

# Based on the elbow/silhouette method, we decided the number of clusters wil be 3

# Step 6: Compute Distance Matrix
# Compute the distance matrix using Euclidean distance
distance_matrix <- dist(scaled_sample, method = "euclidean")

# Step 7: Perform Hierarchical Clustering
# Perform hierarchical clustering using the "ward.D2" method
hc <- hclust(distance_matrix, method = "ward.D2")

# Step 8: Visualize the Dendrogram
# Plot the dendrogram to visualize the hierarchical clustering
plot(hc, labels = FALSE, main = "Dendrogram of Hotel Data Sample", xlab = "", sub = "", ylab = "Height")

# Optional: Use a more advanced dendrogram visualization
dend <- as.dendrogram(hc)
dend_colored <- color_branches(dend, k = 3)  # Color branches based on 3 clusters
plot(dend_colored)

# Step 9: Cut the Dendrogram and Assign Clusters
# Choose the number of clusters based on the dendrogram
# Cut the tree into 3 clusters 
clusters <- cutree(hc, k = 3)

# Visualize the clusters
fviz_cluster(list(data = scaled_sample, cluster = clusters), geom = "point")

# Step 10: Analyze Clusters
# Add the cluster labels back to the sampled dataset
hotel_sample$cluster <- clusters

# Summary of clusters
summary_by_cluster <- hotel_sample %>%
  group_by(cluster) %>%
  summarise(
    avg_lead_time = mean(lead_time),
    avg_adr = mean(adr),
    avg_special_requests = mean(total_of_special_requests),
    avg_deposit_type = mean(deposit_type),
    avg_customer_type = mean(customer_type),
    avg_market_segment = mean(market_segment)
  )

print(summary_by_cluster)




colnames(hotel_data)

if (!is.factor(hotel_data$customer_type)) {
  hotel_data$customer_type <- as.factor(hotel_data$customer_type)
}

if (!is.factor(hotel_data$market_segment)) {
  hotel_data$market_segment <- as.factor(hotel_data$market_segment)
}

levels_customer_type <- levels(hotel_data$customer_type)
levels_market_segment <- levels(hotel_data$market_segment)
print(levels_customer_type)
print(levels_market_segment)


# Step 2: Map the numeric values back to the original factor levels for hotel_sample
hotel_sample <- hotel_sample %>%
  mutate(
    # Mapping customer_type
    customer_type_mapped = case_when(
      customer_type == 1 ~ levels_customer_type[1],
      customer_type == 2 ~ levels_customer_type[2],
      customer_type == 3 ~ levels_customer_type[3],
      customer_type == 4 ~ levels_customer_type[4],
      TRUE ~ "Unknown"
    ),
    # Mapping market_segment
    market_segment_mapped = case_when(
      market_segment == 1 ~ levels_market_segment[1],
      market_segment == 2 ~ levels_market_segment[2],
      market_segment == 3 ~ levels_market_segment[3],
      market_segment == 4 ~ levels_market_segment[4],
      market_segment == 5 ~ levels_market_segment[5],
      market_segment == 6 ~ levels_market_segment[6],
      market_segment == 7 ~ levels_market_segment[7],
      TRUE ~ "Unknown"
    )
  )

# Step 3: View the new mapped columns
print(hotel_sample %>% select(cluster, customer_type, customer_type_mapped, market_segment, market_segment_mapped))




# Update the dataset with categorical mappings
hotel_sample$customer_type_mapped <- factor(hotel_sample$customer_type, 
                                            levels = c(1, 2, 3, 4), 
                                            labels = c("Contract", "Group", "Transient", "Transient-Party"))

hotel_sample$market_segment_mapped <- factor(hotel_sample$market_segment, 
                                             levels = c(1, 2, 3, 4, 5, 6, 7), 
                                             labels = c("Complementary", "Corporate", "Direct", "Groups", 
                                                        "Offline TA/TO", "Online TA", "Transient"))




# Enhanced Cluster Plot with customer_type and market_segment
fviz_cluster(list(data = scaled_sample, cluster = clusters), geom = "point", 
             ellipse.type = "convex", show.clust.cent = TRUE, 
             main = "Cluster Plot with Mapped Categorical Variables",
             ggtheme = theme_minimal()) +
  labs(title = "Clusters with Mapped Customer Type and Market Segment")


# Function to find the most common value in each cluster
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

# Enhanced summary table with categorical mappings
summary_by_cluster_mapped <- hotel_sample %>%
  group_by(cluster) %>%
  summarise(
    avg_lead_time = mean(lead_time),
    avg_adr = mean(adr),
    avg_special_requests = mean(total_of_special_requests),
    avg_deposit_type = mean(deposit_type),
    most_common_customer_type = Mode(customer_type_mapped),
    most_common_market_segment = Mode(market_segment_mapped)
  )

# Print the summary
print(summary_by_cluster_mapped)





# Visualizing the distribution of customer_type within each cluster
ggplot(hotel_sample, aes(x = cluster, fill = customer_type_mapped)) +
  geom_bar(position = "dodge") +
  labs(title = "Customer Type Distribution by Cluster", x = "Cluster", y = "Count")

# Visualizing the distribution of market_segment within each cluster
ggplot(hotel_sample, aes(x = cluster, fill = market_segment_mapped)) +
  geom_bar(position = "dodge") +
  labs(title = "Market Segment Distribution by Cluster", x = "Cluster", y = "Count")






# Define more descriptive names for each cluster
cluster_labels <- c(
  "1" = "Short Lead Time & High ADR",
  "2" = "Moderate Lead Time & Medium ADR",
  "3" = "Long Lead Time & Low ADR"
)

# Visualizing the distribution of customer_type within each cluster with new labels
ggplot(hotel_sample, aes(x = factor(cluster, levels = names(cluster_labels), labels = cluster_labels), fill = customer_type_mapped)) +
  geom_bar(position = "dodge") +
  labs(title = "Customer Type Distribution by Cluster", x = "Cluster", y = "Count") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Visualizing the distribution of market_segment within each cluster with new labels
ggplot(hotel_sample, aes(x = factor(cluster, levels = names(cluster_labels), labels = cluster_labels), fill = market_segment_mapped)) +
  geom_bar(position = "dodge") +
  labs(title = "Market Segment Distribution by Cluster", x = "Cluster", y = "Count") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

