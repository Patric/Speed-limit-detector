from video import *
from analyser import *
import keyboard
import imutils



cap = video("resource.mp4").capture()


if cap is not False:
    frame = cap.read()
    pause = False
    analyser = analyser(cap.read()[1])
    #analyser.showFAWTrackbars()
  
    while(cap.isOpened()): 
        d = analyser.getFAWTrackbarValues()

        if keyboard.is_pressed("d"):
            pause = True
        if keyboard.is_pressed("s"):
            pause = False

        #image processing
        if pause:
            pass
        else:
            ret, frame = cap.read()
            processed_frame = analyser.preprocessing(frame)

        #feature extraction
        only_red = analyser.rMask(processed_frame)
        detected_circles = analyser.houghCircles(only_red)
        analyser.cutSuspect(detected_circles)
        analyser.showFrames()
       
        if cv2.waitKey(d['Delay']) & 0xFF == ord('q'):
            break

   
    cap.release()
    cv2.destroyAllWindows()