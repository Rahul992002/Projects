import cv2

# Method to draw boundary around the detected feature
def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    # Convert image to gray-scale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect features in the gray-scale image
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    # Draw rectangle around the feature and label it
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        # Adjust label position to avoid being cut off
        label_y = y - 10 if y - 10 > 10 else y + h + 10  # Position above the rectangle
        cv2.putText(img, text, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return coords

# Method to detect the features
def detect(img, faceCascade, eyeCascade, noseCascade, mouthCascade):
    color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0), "white": (255, 255, 255)}
    coords = draw_boundary(img, faceCascade, 1.1, 10, color['blue'], "Face")
    
    # If feature is detected, draw other features
    if len(coords) == 4:
        # Updating region of interest by cropping image
        roi_img = img[coords[1]:coords[1] + coords[3], coords[0]:coords[0] + coords[2]]
        
        # Passing roi, classifier, scaling factor, Minimum neighbours, color, label text
        draw_boundary(roi_img, eyeCascade, 1.1, 12, color['red'], "Eye")
        draw_boundary(roi_img, noseCascade, 1.1, 4, color['green'], "Nose")
        draw_boundary(roi_img, mouthCascade, 1.1, 20, color['white'], "Mouth")
    
    return img

# Load classifiers
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyesCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
noseCascade = cv2.CascadeClassifier('Nariz.xml')  # Ensure this file exists
mouthCascade = cv2.CascadeClassifier('Mouth.xml')  # Ensure this file exists

# Capturing real-time video stream. 0 for built-in web-cams, -1 for external web-cams
video_capture = cv2.VideoCapture(0)

while True:
    # Read image from video stream
    ret, img = video_capture.read()  # Changed to ret for error checking
    if not ret:
        print("Failed to grab frame")
        break

    # Call method we defined above
    img = detect(img, faceCascade, eyesCascade, noseCascade, mouthCascade)

    # Show processed image in a new window
    cv2.imshow("Face Detection", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam
video_capture.release()
# Destroy output window
cv2.destroyAllWindows()
