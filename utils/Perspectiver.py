import cv2 
import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.cluster import KMeans, DBSCAN 
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score 

import optuna
from joblib import Parallel, delayed
import torch

import concurrent.futures

class Perspectiver:

    @staticmethod
    def plotComparison (imageBefore, imageAfter, titleBefore: str = "Before", titleAfter: str = "After"):

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(imageBefore)
        plt.title(titleBefore)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(imageAfter)
        plt.title(titleAfter)
        plt.axis('off')

        plt.show()

    @staticmethod
    def imageScaling(image: np.array, scale: float) -> np.array:
        if not isinstance(image, np.ndarray):
            raise TypeError("The input image must be a NumPy array.")
        if scale <= 0:
            raise ValueError("The scaling factor must be greater than 0.")
        
        new_dim = (int(image.shape[1] * scale), int(image.shape[0] * scale))
        return cv2.resize(image, new_dim, interpolation=cv2.INTER_LINEAR)
    
    @staticmethod
    def meanShift(image: np.array, sp: float, sr: float) -> np.array:
        """
        Perform mean shift filtering on an image.

        Args:
            image (np.array): Input image as a NumPy array (should be in BGR format if using OpenCV).
            sp (float): Spatial window radius. Higher values mean a larger area for smoothing.
            sr (float): Color window radius. Higher values mean more color smoothing.
        Returns:
            np.array: Image after applying mean shift filtering.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("The input image must be a NumPy array.")
        if sp <= 0 or sr <= 0:
            raise ValueError("Both 'sp' and 'sr' must be positive values.")
        # Apply mean shift filtering
        filtered_image = cv2.pyrMeanShiftFiltering(image, sp, sr)
        return filtered_image

    @staticmethod
    def knnFiltering(image: np.array, d: int, sigma_color: float, sigma_space: float) -> np.array:
        """
        Perform K-Nearest Neighbors (KNN) filtering on an image.

        Args:
            image (np.array): Input image as a NumPy array (BGR format for OpenCV).
            d (int): Diameter of each pixel neighborhood for filtering. 
                    If negative, it is calculated from sigma_space.
            sigma_color (float): Filter strength for color. Larger values mean stronger smoothing.
            sigma_space (float): Filter strength for spatial distance. Larger values mean smoothing across a larger area.

        Returns:
            np.array: Image after applying KNN filtering.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("The input image must be a NumPy array.")
        if d < 0:
            raise ValueError("'d' (neighborhood diameter) must be a non-negative integer or -1.")
        if sigma_color <= 0 or sigma_space <= 0:
            raise ValueError("'sigma_color' and 'sigma_space' must be positive values.")
        
        # Apply KNN filtering
        filtered_image = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        return filtered_image
    
    @staticmethod
    def kmeansClustering(image: np.array, k: int, max_iter: int = 100, epsilon: float = 0.2, attempts: int = 10) -> np.array:
        """
        Perform KMeans clustering on an image for segmentation.

        Args:
            image (np.array): Input image as a NumPy array (BGR format for OpenCV).
            k (int): Number of clusters (K).
            max_iter (int): Maximum number of iterations for KMeans (default: 100).
            epsilon (float): Convergence criteria threshold (default: 0.2).
            attempts (int): Number of attempts for KMeans algorithm to find the best clusters (default: 10).

        Returns:
            np.array: Image with pixels replaced by their cluster centers.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("The input image must be a NumPy array.")
        if k <= 0:
            raise ValueError("'k' must be a positive integer.")
        
        # Reshape the image to a 2D array of pixels
        pixel_values = image.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)

        # Define criteria for KMeans
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)

        # Perform KMeans clustering
        _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)

        # Convert centers to uint8 and replace pixels with cluster centers
        centers = np.uint8(centers)
        segmented_image = centers[labels.flatten()]
        segmented_image = segmented_image.reshape(image.shape)
        
        return segmented_image
    
    @staticmethod
    def dbscanClustering(image: np.array, eps: float, min_samples: int) -> np.array:
        """
        Perform DBSCAN clustering on an image for segmentation.

        Args:
            image (np.array): Input image as a NumPy array (BGR format for OpenCV).
            eps (float): Maximum distance between two samples to be considered as neighbors.
            min_samples (int): Minimum number of samples in a neighborhood to be considered a core point.

        Returns:
            np.array: Image with pixels replaced by their cluster centers.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("The input image must be a NumPy array.")
        if eps <= 0:
            raise ValueError("'eps' must be a positive value.")
        if min_samples <= 0:
            raise ValueError("'min_samples' must be a positive integer.")
        
        # Reshape the image to a 2D array of pixels
        pixel_values = image.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)

        # Apply DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(pixel_values)

        # Replace outliers (-1 label) with black color
        unique_labels = np.unique(labels)
        num_clusters = len(unique_labels[unique_labels >= 0])
        cluster_colors = np.random.randint(0, 255, size=(num_clusters, 3), dtype=np.uint8)
        
        segmented_image = np.zeros_like(pixel_values, dtype=np.uint8)
        for label in range(num_clusters):
            segmented_image[labels == label] = cluster_colors[label]
        segmented_image[labels == -1] = [0, 0, 0]  # Outliers as black

        segmented_image = segmented_image.reshape(image.shape)
        return segmented_image
    
    @staticmethod
    def rgb_to_grayscale(image):
        """
        Convert an RGB image (as a numpy array) to grayscale using the luminosity method.

        Args:
            image (np.array): A 3D numpy array representing an RGB image (height x width x 3).

        Returns:
            np.array: A 2D numpy array representing the grayscale image (height x width).
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a 3D numpy array with shape (height, width, 3).")

        # Apply the luminosity method
        grayscale = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
        return grayscale

    @staticmethod
    def evaluate_clustering(original_image, clustered_image):
        """
        Evaluate the quality of clustering using internal evaluation metrics.

        Args:
            original_image (PIL.Image): The original RGB image.
            clustered_image (PIL.Image): The clustered RGB image.

        Returns:
            dict: A dictionary containing the Silhouette Score, Davies-Bouldin Index,
                and Calinski-Harabasz Index.
        """
        # Convert images to grayscale
        original_gray = Perspectiver.rgb_to_grayscale(original_image).flatten()
        clustered_gray = Perspectiver.rgb_to_grayscale(clustered_image).flatten()

        # Ensure the images have the same dimensions
        if original_gray.shape != clustered_gray.shape:
            raise ValueError("Original and clustered images must have the same dimensions.")

        # Calculate evaluation metrics
        silhouette = silhouette_score(original_gray.reshape(-1, 1), clustered_gray)
        davies_bouldin = davies_bouldin_score(original_gray.reshape(-1, 1), clustered_gray)
        calinski_harabasz = calinski_harabasz_score(original_gray.reshape(-1, 1), clustered_gray)

        # Return the metrics as a dictionary
        return {
            'Silhouette Score': silhouette,
            'Davies-Bouldin Index': davies_bouldin,
            'Calinski-Harabasz Index': calinski_harabasz,
            'Silhouette Score/n': silhouette/len(np.unique(clustered_image)),
            'Davies-Bouldin Index/n': davies_bouldin/len(np.unique(clustered_image)),
            'Calinski-Harabasz Index/n': calinski_harabasz/len(np.unique(clustered_image))
        }
        
    @staticmethod
    def grayscale_to_rgb(grayscale_image: np.array) -> np.array:
        """
        Converts a grayscale image to an RGB image.

        Args:
            grayscale_image (np.array): Input grayscale image as a NumPy array.

        Returns:
            np.array: RGB image as a NumPy array.
        """
        if not isinstance(grayscale_image, np.ndarray):
            raise TypeError("The input must be a NumPy array.")
        if len(grayscale_image.shape) != 2:
            raise ValueError("The input must be a 2D grayscale image.")
        
        # Convert grayscale to RGB by stacking the grayscale image along the third axis
        rgb_image = np.stack((grayscale_image,)*3, axis=-1)
        return rgb_image
    
    @staticmethod
    def normalize_to_uint8(normalized_image: np.array) -> np.array:
        """
        Converts a normalized image (values between 0 and 1 or any range) to an image with values between 0 and 255.

        Args:
            normalized_image (np.array): Input normalized image as a NumPy array.

        Returns:
            np.array: Image with values scaled to the range 0 to 255.
        """
        if not isinstance(normalized_image, np.ndarray):
            raise TypeError("The input must be a NumPy array.")
        
        # Find the minimum and maximum of the image to scale it
        min_val = np.min(normalized_image)
        max_val = np.max(normalized_image)
        
        if min_val == max_val:
            raise ValueError("The image has no variation (all values are the same).")
        
        # Scale the image to the range 0-255
        scaled_image = (normalized_image - min_val) / (max_val - min_val) * 255.0
        return scaled_image.astype(np.uint8)