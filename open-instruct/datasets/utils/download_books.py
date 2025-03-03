import kagglehub

# Download latest version
path = kagglehub.dataset_download("elvinrustam/books-dataset")

print("Path to dataset files:", path)