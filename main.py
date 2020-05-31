from video import *
from roadLimitDetector import *
import keyboard
import imutils



cap = video("resource.mp4").capture()


if cap is not False:
    frame = cap.read()
    pause = False
    detector = roadLimitDetector(cap.read()[1])
    detector.createTrackbars("Parameters")
  
    while(cap.isOpened()): 
        d = detector.getTrackbarValues()

        if keyboard.is_pressed("d"):
            pause = True
        if keyboard.is_pressed("s"):
            pause = False

        #image processing
        if pause:
            pass
        else:
            ret, frame = cap.read()
            processed_frame = detector.preprocessing(frame)

        #feature extraction
        only_red = detector.rMask(processed_frame)
        detected_circles = detector.houghCircles(only_red)
        detector.markFrames(detected_circles)
        detector.showFrames()
       
        if cv2.waitKey(d['Delay']) & 0xFF == ord('q'):
            break

   
    cap.release()
    cv2.destroyAllWindows()