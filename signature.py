import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt

# TODO add contour detection for enhanced accuracy


def extract_signature(image_path):

    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to smooth the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply median filter to further reduce noise
    filtered = cv2.medianBlur(blurred, 5)

    # erosion
    eroded_image = cv2.erode(filtered,(5, 5),iterations=1)

    # Perform Dilation
    dilated_image = cv2.dilate(eroded_image,(5, 5), iterations=1)

    # Apply thresholding to extract the signature
    _, thresh = cv2.threshold(dilated_image, 170, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (presumably the signature)
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop the signature region from the original image
    signature_crop = image[y:y+h, x:x+w]
    return signature_crop

def match(path1, path2):

    # Path to the image containing the signature
    image_path1 = path1
    image_path2 = path2
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    # turn images to grayscale
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # resize images for comparison
    img1 = cv2.resize(img1, (300, 300))
    img2 = cv2.resize(img2, (300, 300))

    # display both images
    cv2.imshow("One", img1)
    cv2.imshow("Two", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Extract the signature
    extracted_signature1 = extract_signature(image_path1)
    extracted_signature2 = cv2.imread(path2)

    # Resize the images to have the same dimensions
    resized_signature1 = cv2.resize(extracted_signature1, (extracted_signature2.shape[1], extracted_signature2.shape[0]))
    print_resized = cv2.resize(resized_signature1, (300, 300))
    cv2.imshow("OUTPUT", print_resized)
    # cv2.waitKey(0)

    # Calculate the absolute difference between the two signatures
    abs_diff = np.mean(np.abs(resized_signature1.astype(np.float32) - extracted_signature2.astype(np.float32)))
    abs_diff_threshold = 10

    # Compare the two signatures
    if abs_diff <= abs_diff_threshold:
        similarity_value = True
    else:
        similarity_value = False

    print(similarity_value)

    return similarity_value

