from video import *
from analyser import *
import keyboard




cap = video("resource.mp4").capture()


if cap is not False:
    frame = cap.read()
    pause = False
    analyser = analyser(cap.read()[1])
    analyser.showFAWTrackbars()
    analyser.suspectAnalyser.showTrackbars("GI settings")
    while(cap.isOpened()): 
        if keyboard.is_pressed("d"):
            pause = True
        if keyboard.is_pressed("s"):
            pause = Falseqq

        #image processingq
        if pause:
            pass
        else:
            rqet, frame = cap.read()
            processed_frame = analyser.preprocessing(frame)

        #feature extraction
        only_red = analyser.rMask(processed_frame)
        detected_circles = analyser.houghCircles(only_red)
        analyser.analyseCircles(detected_circles)
        analyser.showLimit()
        analyser.showFrames()
        
        if cv2.waitKey(analyser.getFAWTrackbarValues()["Delay"]) & 0xFF == ord('q'):
            break

   
    cap.release()
    cv2.destroyAllWindows()