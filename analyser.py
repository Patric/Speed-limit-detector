from video import *
from suspectAnalyser import *


def callback(foo):
        pass


class analyser:
    scale = 50
    width = 0
    height = 0
    FAW_settings = 0
    SAW_settings = 0
    frame_analysys_winname = "Disabled"#disabled by default
    suspectAnalyser = suspectAnalyser()

    current_limit = None
    prediction = None

    momentary_predictions = list()

    BGR_frame = 0
    #rmask = 0
    #rmask_copy = 0
    bin_mask = 0
    suspects = list()
    counter = 0

    def __init__(self, frame):
        #reading first frame to get setting for preprocessing
        scale = self.scale
        self.BGR_frame = frame
        self.width = int(frame.shape[1] * scale / 100)
        self.height = int(frame.shape[0] * scale / 100)
        #'Settting name' : (default value, range in trackbar)
        #later weather profile can be loaded from .ini file
        self.FAW_settings = {
            'Gaussian kernel size': (6, 50),
            'Delay': (1, 100),
            'H min': (0, 180),
            'H max': (13, 180),
            'S min': (95, 255),
            'S max': (255, 255),
            'V min': (0, 255),
            'V max': (238, 255),
            'H2 min': (126, 180),
            'H2 max': (180, 180),
            'S2 min': (45, 255),
            'S2 max': (255, 255),
            'V2 min': (0, 255),
            'V2 max': (239, 255),
            'contrast' : (254, 508),
            'brightness' : (254, 508),
            'gamma' : (127, 254),
            'canny param': (int(max(self.width, self.height)/9), 300),
            'center param': (int(max(self.width, self.height)/24), 45),
            'open kernel': (int(max(self.width, self.height)/192), 100),
            'close kernel': (int(max(self.width, self.height)/192), 100),
            'open i': (3, 15),
            'close i': (int(max(self.width, self.height)/192), 15),
            'min rad': (int(max(self.width, self.height)/65), 40),
            'max rad': (int(max(self.width, self.height)/20), 80),
            'min dist': (int(max(self.width, self.height)/320), 20),
            'dp for hough': (12, 100)
        }

    #Private methods
    def __resizeFrame(self, frame, scale):
        try:
            width = int(frame.shape[1] * scale / 100)
            height = int(frame.shape[0] * scale / 100)
            return cv2.resize(frame, (self.width, self.height))
        except Exception as e:
                #couldn't resize or smth
                print(str(e))

    def showFAWTrackbars(self):
        self.frame_analysys_winname = "FAW settings window"
        cv2.namedWindow(self.frame_analysys_winname, cv2.WINDOW_NORMAL)
        for key in self.FAW_settings:
            cv2.createTrackbar(key, self.frame_analysys_winname, self.FAW_settings[key][0], self.FAW_settings[key][1], callback)


    def getFAWTrackbarValues(self):
        d = dict()
        if self.frame_analysys_winname is not "Disabled":
            for key in self.FAW_settings:
                d.update( { key : int(cv2.getTrackbarPos(key, self.frame_analysys_winname)) } )
            d['dp for hough'] = d['dp for hough'] / 10
            return d
        elif self.frame_analysys_winname is "Disabled":
            for key in self.FAW_settings:
                d.update( { key : self.FAW_settings[key][0] } )
            d['dp for hough'] = d['dp for hough'] / 10
            return d

   #Feature extraction

    def preprocessing(self, frame):
        self.BGR_frame = self.__resizeFrame(frame, self.scale)
     
        #eliminating high frequency noise
        blurred = cv2.GaussianBlur(self.BGR_frame, (self.FAW_settings['Gaussian kernel size'][0]*2+1, self.FAW_settings['Gaussian kernel size'][0]*2+1), 0)
        return blurred

    def apply_brightness_contrast(self, frame, brightness = 0, contrast = 0):
        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha_b = (highlight - shadow)/255
            gamma_b = shadow

            buf = cv2.addWeighted(frame, alpha_b, frame, 0, gamma_b)
        else:
            buf = frame.copy()

        if contrast != 0:
            f = 131*(contrast + 127)/(127*(131-contrast))
            alpha_c = f
            gamma_c = 127*(1-f)

            buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

        return buf

    #Used to get settings e.g to write to a file - for later
    def getSettings(self):
        pass
    
    #Slicing colours different than red
    def rMask(self, frame):
        d = self.getFAWTrackbarValues()

        #self.BGR_frame = cv2.addWeighted(self.BGR_clean, d['contrast'] * 0.01, self.BGR_frame, d['brightness'] * 0.01, d['gamma'] - 127)
        
        #frame = self.BGR_frame
        #self.BGR_frame= self.apply_brightness_contrast(frame, d['brightness'] - 254, d['contrast'] - 254)
        frame = cv2.cvtColor(self.BGR_frame, cv2.COLOR_BGR2HSV)

        red_in_range_1 = cv2.inRange(frame, (d['H min'], d['S min'], d['V min']), (d['H max'], d['S max'] ,d['V max']))
        red_in_range_2 = cv2.inRange(frame, (d['H2 min'], d['S2 min'], d['V2 min']), (d['H2 max'], d['S2 max'] ,d['V2 max']))
        red_ranges_combined = cv2.bitwise_or(red_in_range_1, red_in_range_2)

        #self.rmask = cv2.bitwise_and(self.BGR_frame, self.BGR_frame, mask = red_ranges_combined)
        

        red_opened = cv2.morphologyEx(red_ranges_combined, cv2.MORPH_OPEN, (d['open kernel'], d['open kernel']), iterations=d['open i'])
        red_closed = cv2.morphologyEx(red_opened, cv2.MORPH_CLOSE, (d['close kernel'], d['close kernel']), iterations=d['close i'])
        self.bin_mask = red_closed
        self.bin_mask = cv2.cvtColor(self.bin_mask, cv2.COLOR_GRAY2BGR)
        #self.rmask_copy = cv2.bitwise_and(self.BGR_frame, self.BGR_frame, mask = red_closed)
        return red_closed

    #Hough tranfsorm
    def houghCircles(self, bin_frame):
        #inversion for hough
       
        red_closed = bin_frame
        red_closed = cv2.bitwise_not(red_closed)

        #loading setting dictionary 
        d = self.getFAWTrackbarValues()
        
    
        circles = cv2.HoughCircles(bin_frame,cv2.HOUGH_GRADIENT, d['dp for hough'],d['min dist'],
                            param1=d['canny param'],param2=d['center param'],minRadius=d['min rad'],maxRadius=d['max rad'])
    
        return circles

    def __markFrame(self, frame_to_mark, x, y, r, prediction):
        cv2.circle(frame_to_mark, (x, y), r, (0, 255, 0), 2)
        cv2.rectangle(frame_to_mark, (x - 2, y - 2), (x + 2, y + 2), (0, 128, 255), -1)
        # if prediction != None:
        #     cv2.putText(frame_to_mark, f"LIMIT {prediction} km/h", (x - 400, y - 40),
        #             cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

        return frame_to_mark
        
    def showLimit(self):
        
        if self.current_limit != self.prediction and self.prediction != None:
            self.current_limit = self.prediction
        if self.current_limit != None:
            cv2.putText(self.BGR_frame, f"CURRENT SPEED LIMIT: {self.current_limit} km/h",
            (0, 0 + int(self.BGR_frame.shape[0])-20),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    def cropSuspect(self, frame, x, y, r):
        suspect = frame[y-int(1.0*r):y+int(1.0*r), x-int(1.0*r):x+int(1.0*r)]
        return suspect


    def most_frequent(self, numbers): 
        counter = 0
        num = numbers[0]
      
        for i in numbers: 
            curr_frequency = numbers.count(i) 
            if(curr_frequency> counter): 
                counter = curr_frequency 
                num = i 
    
        return num 

    def analyseCircles(self, circles):
        #drawing circles
        self.counter += 1
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            predictions = list()
            for (x, y, r) in circles:
                #acceleration
                #discard close to each other cirles

                #Cutting prohibition sign from the background
                #suspect consists of red mask and a BGR frame
                suspect_bin = self.cropSuspect(self.bin_mask, x, y, r)
                suspect_bgr = self.cropSuspect(self.BGR_frame, x, y, r)
                prediction = self.suspectAnalyser.analyseSuspect(suspect_bgr, suspect_bin)
                predictions.append(prediction)
       
                #Draw on redmask
            self.prediction = self.most_frequent(predictions)
            if self.prediction != None:
                self.momentary_predictions.append(self.prediction)
                #print(f"CURRENT PRED SET:{self.momentary_predictions}")
                self.prediction = self.most_frequent(self.momentary_predictions)
                self.__markFrame(self.bin_mask, x, y, r, prediction)
        #compares output to output from last 120 frames
        if self.counter == 120:
            self.momentary_predictions = list()
            self.counter = 0
            #print("detected circle")#for debug
        
  
   
    def showFrames(self):
        cv2.imshow('only_red', self.bin_mask)
        cv2.imshow('frame', self.BGR_frame)

    