from analyser import *
import imutils
from datetime import datetime
from speedLNetManager import *
import pathlib
def callback(foo):
        pass


class suspectAnalyser:
    suspects = list()
    settings = list()
    width = 0
    height = 0
    cntr = 0
    neuralAnalyser = 0

    def __init__(self):
        d = {
            'Gaussian kernel size': (6, 50),
            'H min': (0, 180),
            'H max': (190, 180),
            'S min': (0, 255),
            'S max': (255, 255),
            'V min': (0, 255),
            'V max': (255, 255),
            'contrast' : (1200, 2000),
            'brightness' : (870, 3000),
            'gamma' : (50, 200),
            't max value' : (255, 255),
            'bsize' : (28, 50),
            'C' : (32, 50),
            'open kernel': (5, 100),
            'close kernel': (5, 100),
            'open i': (10, 60),
            'close i': (10, 60),
            'Cnt thresh' : (1240, 10000),
            'Cnt thresh 2' : (4000, 10000)
        }
        get_inner_settings = {"GI settings" : d}
        self.settings.append(get_inner_settings)

        batchsize = 1
        iteration = 4
        PATH = './sld_my_net.pth'
        self.neuralAnalyser = speedLNetManager(batchsize)
        


        self.neuralAnalyser.loadNet(PATH)
        
        # for param in self.neuralAnalyser.net.features.parameters():
        #     param.requires_grad = False

        self.neuralAnalyser.net.eval()

       

    def showTrackbars(self, setting_to_use):
        cv2.namedWindow(setting_to_use, cv2.WINDOW_NORMAL)
        for setting in self.settings:
            d = setting.get(setting_to_use)
            if d:
                for key in d:
                    cv2.createTrackbar(key, setting_to_use, d[key][0], d[key][1], callback)


    def getTrackbarValues(self, setting_to_use):
        d = dict()
        for setting in self.settings:
                try:
                    for key in setting.get(setting_to_use):
                        d.update( { key : int(cv2.getTrackbarPos(key, setting_to_use)) } )
                    return d
                except Exception as e:
                    for key in setting.get(setting_to_use):
                        d.update( { key : setting.get(setting_to_use)[key][0] } )
                    return d

    def generateDataSet(self, data):
        cv2.imwrite(f'dataset_32x32/suspect_{self.cntr}.jpg', data) 
        self.cntr += 1


    def getCipher(self, suspect_bin, suspect_bgr):
        
        d = self.getTrackbarValues("GI settings")
        suspect_bgr = cv2.addWeighted(suspect_bgr, d['contrast']*0.001, suspect_bgr, d['brightness'] *0.001, d['gamma']-100)
        suspect_hsv = cv2.cvtColor(suspect_bgr, cv2.COLOR_BGR2HSV)
        #eliminating high frequency noise
        suspect_hsv = cv2.GaussianBlur(suspect_hsv, (d['Gaussian kernel size']*2+1, d['Gaussian kernel size']*2+1), 0)
        cut_black = cv2.inRange(suspect_hsv, (d['H min'], d['S min'], d['V min']), (d['H max'], d['S max'] ,d['V max']))
        bmask = cv2.bitwise_and(suspect_bgr, suspect_bgr, mask = cut_black)
        bmask = cv2.cvtColor(bmask, cv2.COLOR_BGR2GRAY)
        threshold = cv2.adaptiveThreshold(bmask, maxValue = d['t max value'], adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType = cv2.THRESH_BINARY, blockSize = d['bsize']*2 + 1, C = d['C'] - 25)
        opened = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, (d['open kernel'], d['open kernel']), iterations=d['open i'])
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, (d['close kernel'], d['close kernel']), iterations=d['close i'])
    
        
        cnts, hierarchy = cv2.findContours(closed.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #cnts = imutils.grab_contours(cnts)
        
        hierarchy = hierarchy[0]
        view = closed
     
        cv2.imshow("closed", closed)
        #loop over the contours
        predictions = list()
        for component in zip(cnts, hierarchy):
            c = component[0]

            if cv2.contourArea(c) > d['Cnt thresh'] and  cv2.contourArea(c) < d['Cnt thresh 2']:
                # M = cv2.moments(c)
                # cX = int(M["m10"] / M["m00"])
                # cY = int(M["m01"] / M["m00"])
                # draw the contour and center of the shape on the image
                
                #cv2.circle(view, (cX, cY), 7, (0, 255, 0), -1)
                #cv2.putText(view, f"center", (cX - 20, cY - 20),
                #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                #cv2.imshow('cnts', view)
                cipher = np.zeros(view.shape).astype(view.dtype)


                cipher = cv2.fillPoly(cipher, [c], 255)
                cipher = cv2.bitwise_not(cipher)
                
                # result2 = cv2.cvtColor(cipher, cv2.COLOR_GRAY2BGR)
                # cv2.drawContours(result2, [c], -1, (0, 255, 0), 2)
                # cv2.imshow('suspect', result2)

                # cv2.waitKey(0)
                x,y,w,h = cv2.boundingRect(c)
               

             
                cipher = cipher[y:y+h, x:x+h]
        
                kernel = np.ones((3,3),np.uint8)
                cipher = cv2.dilate(cipher,kernel,iterations = 4)
                
         
                toEvaluate = cv2.resize(cipher, (32, 32))
                toEvaluate = cv2.bitwise_not(toEvaluate)
                #self.generateDataSet(toEvaluate)
                prediction = self.neuralAnalyser.evaluate(toEvaluate)
                cv2.imshow("eval", toEvaluate)
                # print(prediction)
                # cv2.waitKey(0)
                
                if prediction != "unidentified":
                    predictions.append(prediction)
       
                #print(label)
                #x,y,w,h = cv2.boundingRect(c)
                #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        #
        #return closed to view dataset
        
       
        # if prediction != "unidentified" and prediction != None:
        #             cv2.putText(result, f"LIMIT {prediction} km/h", (x - 400, y - 40),
        #             cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                  
        # image = toEvaluate
        # cv2.imshow("LOL", image)
        # image = image[np.newaxis,:]
        # image = image[np.newaxis,:]
     
        # image = torch.from_numpy(image)
        # image = image.float()
        # classes1 = ('30', '40', '70', 'unidenqtified')
        # image.to(self.neuralAnalyser.cuda)

        # output = self.neuralAnalyser.net(image)
        # print(torch.argmax(output))
        #probs, classes = output.topk(1, dim=1)

        #print("The model is ", probs.item()*100, "% certain that the image has a predicted class of ", classes1[classes.item()] )
      
       
        #toEvaluate = cv2.bitwise_not(toEvaluate)
    
        return predictions

    def analyseCiphers(self, predictions):
        print(predictions)
        output = list()
        str_o = ""
        for prediction in predictions:
            if prediction != '0':
                output.insert(0, prediction)
            else:
                output.append(prediction)
        for e in output:
            str_o += e
        return str_o


  

    def analyseSuspect(self, suspect_bgr, suspect_bin):
        #40px is minimum value for a shape to be taken into consideration. Used to avoid zero sizes after int rounding
      
        if(suspect_bin.shape[0] > 40 and suspect_bin.shape[0] > 40):
            self.width = suspect_bin.shape[0]*5
            self.height = suspect_bin.shape[1]*5
            suspect_bin = cv2.resize(suspect_bin, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
            suspect_bgr = cv2.resize(suspect_bgr, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
            ciphers = self.getCipher(suspect_bin, suspect_bgr)
            if len(ciphers) > 0:
                number = self.analyseCiphers(ciphers)
                if int(number) % 5 == 0 and int(number) <= 140 and int(number) > 0:
                #cv2.imwrite(f"suspects/suspect{datetime.now()}.jpg", suspect)
                    return number
            # try:
      
            # except Exception as e:
            #     #couldn't resize or smth
            #     print(str(e))

           
