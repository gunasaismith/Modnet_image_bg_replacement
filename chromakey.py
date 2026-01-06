import cv2
import numpy as np

# === Step 1: Load image ===
input_path = 'subject.jpg'  # Update if needed
image = cv2.imread(input_path)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# === Step 2: Create green mask ===
lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])
green_mask = cv2.inRange(hsv, lower_green, upper_green)

# === Step 3: Clean mask using morphology ===
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

# === Step 4: Invert mask to get subject ===
subject_mask = cv2.bitwise_not(green_mask)

# === Step 5: Create RGBA output ===
b, g, r = cv2.split(image)
rgba = cv2.merge([b, g, r, subject_mask])

# === Step 6: Optional green spill suppression ===
def suppress_green_spill(rgba_img):
    b, g, r, a = cv2.split(rgba_img)
    # Where green is dominant, neutralize it
    green_spill = (g > r) & (g > b) & (a > 0)
    g[green_spill] = ((r[green_spill].astype(np.uint16) + b[green_spill].astype(np.uint16)) // 2).astype(np.uint8)
    return cv2.merge([b, g, r, a])

rgba_cleaned = suppress_green_spill(rgba)

# === Step 7: Save result ===
output_path = 'subject_no_green.png'
cv2.imwrite(output_path, rgba_cleaned)
print(f"Saved transparent image to: {output_path}")
