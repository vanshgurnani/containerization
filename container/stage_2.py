import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load data from Excel file
file_path = './split_data.xlsx'  # Replace with your file path
xl = pd.ExcelFile(file_path)

# Lists to store values for plotting
n_clusters_values = []
accuracy_values = []

# Process each sheet in the Excel file
for sheet_name in xl.sheet_names:
    # Read data from the current sheet
    df = xl.parse(sheet_name)

    # Extract 'POD' column data
    pod_data = df['POD']

    # Perform TF-IDF vectorization on the 'POD' data
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(pod_data)

    # Evaluate clustering accuracy using silhouette score
    max_clusters = 5  # Set the maximum number of clusters to 5
    silhouette_scores = []

    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X)
        silhouette = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(silhouette)

    # Find the optimal number of clusters based on the highest silhouette score
    optimal_n_clusters = silhouette_scores.index(max(silhouette_scores)) + 2  # Add 2 because range starts from 2

    # Store values for plotting
    n_clusters_values.append(range(2, max_clusters + 1))
    accuracy_values.append(silhouette_scores)

    # Print the optimal number of clusters for the current sheet
    print(f"Optimal number of clusters for sheet '{sheet_name}': {optimal_n_clusters}")

# Plot the line graph
for i, sheet_name in enumerate(xl.sheet_names):
    plt.plot(range(2, max_clusters + 1), accuracy_values[i], marker='o', label=sheet_name)

plt.title('Number of Clusters vs Accuracy')
plt.xlabel('Number of Clusters')
plt.ylabel('Accuracy (Silhouette Score)')
plt.legend()
plt.show()
