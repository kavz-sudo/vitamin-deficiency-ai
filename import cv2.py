import cv2

# Open the default camera (usually the first one)
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

# Read one frame
ret, frame = camera.read()

if ret:
    # Display the image
    cv2.imshow("Captured Image", frame)

    # Save the image
    cv2.imwrite("captured_image.jpg", frame)
    print("Image saved as 'captured_image.jpg'")

    # Wait until a key is pressed
    cv2.waitKey(0)
else:
    print("Failed to capture image.")

# Release the camera
camera.release()
cv2.destroyAllWindows()
