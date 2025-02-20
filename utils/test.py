import cv2

cap = cv2.VideoCapture(1)

# Create the background subtractor object
fgbg = cv2.createBackgroundSubtractorMOG2()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction to detect motion
    fgmask = fgbg.apply(frame)

    # Display the foreground mask (highlight moving objects)
    cv2.imshow('Foreground Mask', fgmask)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
