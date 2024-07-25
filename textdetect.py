import cv2
import numpy as np
import matplotlib.pyplot as plt
import easyocr

def preprocess_image(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Check if the image was read successfully
    if img is None:
        print("Failed to read image. Please check the file path and integrity.")
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to get a binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Remove noise using morphological operations
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # Remove horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    detect_horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts_horizontal = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_horizontal = cnts_horizontal[0] if len(cnts_horizontal) == 2 else cnts_horizontal[1]
    for c in cnts_horizontal:
        cv2.drawContours(binary, [c], -1, (0, 0, 0), 2)

    # Remove vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    detect_vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts_vertical = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_vertical = cnts_vertical[0] if len(cnts_vertical) == 2 else cnts_vertical[1]
    for c in cnts_vertical:
        cv2.drawContours(binary, [c], -1, (0, 0, 0), 2)

    # Optional: Apply Gaussian Blur to smooth edges
    binary = cv2.GaussianBlur(binary, (5, 5), 0)

    return binary

def display_image(img, title="Processed Image"):
    # Display the processed image using matplotlib
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')  # Hide axis
    plt.show()

def read_text_from_image(img):
    # Initialize the EasyOCR reader
    reader = easyocr.Reader(['en'])

    # Read text from the processed image
    results = reader.readtext(img)
    
    # Print the text
    for (bbox, text, prob) in results:
        print(f"Text: {text} (Probability: {prob:.2f})")

def main():
    # Path to your hotel bill image
    image_path = 'opencv\hi.jpg'

    # Preprocess the image
    processed_img = preprocess_image(image_path)

    if processed_img is not None:
        # Display the processed image
        display_image(processed_img)

        # Read and print text from the image
        read_text_from_image(processed_img)

if __name__ == "__main__":
    main()
