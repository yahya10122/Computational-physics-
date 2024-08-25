import cv2 as cv
import numpy as np
from numba import njit
import math
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import os
import time


@njit
def calculate_ellipse_area(cx, cy, a, b, angle):
    semi_major_axis = a / 2
    semi_minor_axis = b / 2
    area = math.pi * semi_major_axis * semi_minor_axis
    return area, angle


@njit
def calculate_correlation(cx, cy, theta, average_angle, r_values):
    c_values = np.zeros(len(r_values))
    vec_diff = theta - average_angle
    for k, r in enumerate(r_values):
        distances = np.sqrt((cx[:, None] - cx[None, :]) ** 2 + (cy[:, None] - cy[None, :]) ** 2)
        mask = np.abs(distances - r) < 0.1
        a = vec_diff[:, None] * vec_diff[None, :] * mask
        mask_sum = np.sum(mask)
        if mask_sum > 0:
            c_values[k] = np.sum(a) / mask_sum
        else:
            c_values[k] = 0  # or np.nan if you prefer

    c_value_at_r_zero = c_values[0]
    if c_value_at_r_zero != 0:
        c_values /= c_value_at_r_zero
    return c_values


def gaussian_filter_multiscale_retinex(image: np.ndarray, sigmas: list, weights: list) -> np.ndarray:
    img32 = image.astype('float32') / 255
    img32_log = np.log1p(img32)

    msr = np.zeros(image.shape, np.float32)
    for sigma, weight in zip(sigmas, weights):
        blur = cv.GaussianBlur(img32, ksize=(0, 0), sigmaX=sigma)
        blur_log = np.log1p(blur)
        msr += (img32_log - blur_log) * weight

    msr /= sum(weights)
    return cv.normalize(msr, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)


def process_image(img_path):
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    rtnx = gaussian_filter_multiscale_retinex(img, sigmas=[15, 55, 185], weights=[10, 5, 1])
    thresholded = cv.adaptiveThreshold(rtnx, 255, adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       thresholdType=cv.THRESH_BINARY, blockSize=7, C=-7)

    nb_components, output, stats, _ = cv.connectedComponentsWithStats(thresholded, connectivity=8)
    sizes = stats[1:, -1]
    new_img = np.zeros_like(thresholded)
    new_img[np.isin(output, np.where(sizes >= 12)[0] + 1)] = 255

    numLabels, labels, stats, centroids = cv.connectedComponentsWithStats(new_img)

    x, y, angles, ellipse_params = [], [], [], []
    for i in range(1, numLabels):
        componentMask = (labels == i).astype('uint8')
        contours, _ = cv.findContours(componentMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        if len(contours) > 0 and len(contours[0]) >= 5:
            ellipse = cv.fitEllipse(contours[0])
            area, angle = calculate_ellipse_area(*ellipse[0], *ellipse[1], ellipse[2])
            if area < 250:
                (cx, cy), (a, b), angle = ellipse
                x.append(cx)
                y.append(cy)
                angles.append(angle)
                ellipse_params.append(ellipse)

    coordinates = np.column_stack((x, y))
    dbscan = DBSCAN(eps=40, min_samples=20)
    labels = dbscan.fit_predict(coordinates)

    cluster_angles = {label: [] for label in set(labels) if label != -1}
    cluster_coordinates = {label: [] for label in set(labels) if label != -1}

    for i, label in enumerate(labels):
        if label != -1:
            cluster_angles[label].append(angles[i])
            cluster_coordinates[label].append(coordinates[i])

    filtered_cluster_coordinates = {}
    filtered_cluster_angles = {}

    for label in cluster_angles:
        angles_array = np.array(cluster_angles[label])
        angle_diff = np.abs(angles_array[:, None] - angles_array[None, :])
        mask = np.any(angle_diff <= 20, axis=1)
        filtered_cluster_coordinates[label] = np.array(cluster_coordinates[label])[mask]
        filtered_cluster_angles[label] = angles_array[mask]

    max_contours_label = max(filtered_cluster_coordinates, key=lambda k: len(filtered_cluster_coordinates[k]))

    if max_contours_label is not None:
        coords = np.array(filtered_cluster_coordinates[max_contours_label])
        theta = np.array(filtered_cluster_angles[max_contours_label])
        cy, cx = coords[:, 1], coords[:, 0]

        average_angle = np.mean(theta)
        r_values = np.arange(0, 150, 0.01)

        # Ensure all inputs are NumPy arrays
        cx = np.array(cx)
        cy = np.array(cy)
        theta = np.array(theta)
        r_values = np.array(r_values)

        c_values = calculate_correlation(cx, cy, theta, average_angle, r_values)

        return r_values, c_values

    return None, None


if __name__ == "__main__":
    start_time = time.time()

    folder_path = "/Users/yahya2/Desktop/Original image, frame by frame copy/"
    image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path)
                   if filename.lower().endswith((".jpg", ".png", ".jpeg"))]

    r_values_all = []
    c_values_all = []

    for img_path in image_paths:
        r_values, c_values = process_image(img_path)
        if r_values is not None and c_values is not None:
            r_values_all.extend(r_values)
            c_values_all.extend(c_values)

    r_values_all = np.array(r_values_all)
    c_values_all = np.array(c_values_all)

    unique_r, indices = np.unique(r_values_all, return_inverse=True)
    c_values_grouped = np.array([np.mean(c_values_all[indices == i]) for i in range(len(unique_r))])

    plt.figure(figsize=(10, 6))
    plt.scatter(unique_r, c_values_grouped, label='Averaged data')
    plt.xlabel('r')
    plt.ylabel('Normalized c')
    plt.legend()
    plt.grid(True)
    plt.show()

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")