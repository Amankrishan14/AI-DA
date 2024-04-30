import cv2
import easyocr
import numpy as np
import time
reader = easyocr.Reader(['en'],gpu=False)
KPS = 3  # Target Keyframes Per Second
cap = cv2.VideoCapture("videoblocks-214z_014_re_ve0dt3__b65dc426dc0a34e0676e2c347df0b06a__P360.mp4")

#to get frame from a particular point
fps = round(cap.get(cv2.CAP_PROP_FPS))
print('frames per second =',fps)

hop = round(fps / KPS)
curr_frame = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    if curr_frame % hop == 0:
        # Perform OCR on the frame
        result = reader.readtext(frame)

        # Display the frame with bounding boxes around the detected text
        for detection in result:
            points = detection[0]
            points = np.array(points, dtype=np.int32)  # Convert to NumPy array
            points = points.reshape((-1, 1, 2))
            frame = cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)

        # Display the resulting frame
        cv2.imshow("Text Extraction from Video", frame)
        print(result)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    curr_frame += 1
cap.release()
cv2.destroyAllWindows()