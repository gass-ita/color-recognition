import cv2
import cv2.aruco as aruco

cap = cv2.VideoCapture(0)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
aruco_params = aruco.DetectorParameters()
aruco_detector = aruco.ArucoDetector(aruco_dict, aruco_params)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # The new detector returns a tuple
    corners, ids, rejected = aruco_detector.detectMarkers(gray)
    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)
        for i, corner in enumerate(corners):
            c = corner[0]
            cx = int(c[:,0].mean())
            cy = int(c[:,1].mean())
            print(f"Phone marker at: {cx}, {cy} (ID {ids[i][0]})")
    cv2.imshow('Aruco Marker Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()