from video import *
from analyser import *
import keyboard




cap = video("resource.mp4").capture()


if cap is not False:
    frame = cap.read()
    pause = False
    analyser = analyser(cap.read()[1])
    #analyser.showFAWTrackbars()
    analyser.suspectAnalyser.showTrackbars("GI settings")
    while(cap.isOpened()): 
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

        #feature extractions
        only_red = analyser.rMask(processed_frame)
        detected_circles = analyser.houghCircles(only_red)
        analyser.analyseCircles(detected_circles)
        analyser.showFrames()
        
        if cv2.waitKey(analyser.getFAWTrackbarValues()["Delay"]) & 0xFF == ord('q'):
            break

   
    cap.release()
    cv2.destroyAllWindows()