from flask import Flask, redirect, url_for, request, render_template

import os

import numpy as np
import time
import cv2
import math
from werkzeug.utils import secure_filename



app = Flask(__name__) 


@app.route('/image/<name>') 
def image(name):
	
	print("IMAGE")
	labelsPath = "./coco.names"
	LABELS = open(labelsPath).read().strip().split("\n")

	np.random.seed(42)
	COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
		dtype="uint8")

	weightsPath = "./yolov3-tiny.weights"
	configPath = "./yolov3-tiny.cfg"

	net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
	imagepath =	os.path.join('E:/college/major project help/major project api/major project/image/images/', name.filename)
	image =cv2.imread(imagepath)
	(H, W) = image.shape[:2]
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()
	print("Frame Prediction Time : {:.6f} seconds".format(end - start))
	print("layeroutputs:")
	print(layerOutputs)
	boxes = []
	confidences = []
	classIDs = []
	for output in layerOutputs:
	    for detection in output:
	        scores = detection[5:]
	        classID = np.argmax(scores)
	        confidence = scores[classID]
	        if confidence > 0.5 and classID == 0:
	            #pdb.set_trace()
	            box = detection[0:4] * np.array([W, H, W, H])
	            (centerX, centerY, width, height) = box.astype("int")
	            x = int(centerX - (width / 2))
	            y = int(centerY - (height / 2))
	            boxes.append([x, y, int(width), int(height)])
	            confidences.append(float(confidence))
	            classIDs.append(classID)
	print("box:")
	print(boxes)
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.3)
	ind = []
	for i in range(0,len(classIDs)):
	    if(classIDs[i]==0):
	        ind.append(i)
	a = []
	b = []
	color = (0,255,0) 
	if len(idxs) > 0:
	        for i in idxs.flatten():
	            (x, y) = (boxes[i][0], boxes[i][1])
	            (w, h) = (boxes[i][2], boxes[i][3])
	            a.append(x)
	            b.append(y)
	            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
	    
	            

	distance=[] 
	nsd = []
	for i in range(0,len(a)-1):
	    for k in range(1,len(a)):
	        if(k==i):
	            break
	        else:
	            x_dist = (a[k] - a[i])
	            y_dist = (b[k] - b[i])
	            d = math.sqrt(x_dist * x_dist + y_dist * y_dist)
	            distance.append(d)
	            if(d<=100.0):
	                nsd.append(i)
	                nsd.append(k)
	            nsd = list(dict.fromkeys(nsd))
	print(nsd)
	color = (0, 0, 255)
	text=""
	for i in nsd:
	    (x, y) = (boxes[i][0], boxes[i][1])
	    (w, h) = (boxes[i][2], boxes[i][3])
	    text="Ok"
	    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
	    text = "Alert"
	    print("Alert")
	    #cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)       
	    cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
	cv2.imshow("Social Distancing Detector", image)
	cv2.imwrite('output.jpg', image)
	cv2.waitKey(0)
	print("completed")
	cv2.destroyAllWindows()
	print(autoit.win_exists("Social Distancing Detector"))
	while(1):
		if not(autoit.win_exists("Social Distancing Detector")):
			break 

	return 'success'

@app.route('/video/<name>') 
def video(name): 
	confid = 0.5
	thresh = 0.5

	vid_path = "./video/videos/video.mp4"
	vidpath =	os.path.join('E:/college/major project help/major project api/major project/video/videos/', name.filename)

	# Calibration needed for each video

	def calibrated_dist(p1, p2):
	    return ((p1[0] - p2[0]) ** 2 + 550 / ((p1[1] + p2[1]) / 2) * (p1[1] - p2[1]) ** 2) ** 0.5


	def isclose(p1, p2):
	    c_d = calibrated_dist(p1, p2)
	    calib = (p1[1] + p2[1]) / 2
	    if 0 < c_d < 0.15 * calib:
	        return 1
	    elif 0 < c_d < 0.2 * calib:
	        return 2
	    else:
	        return 0


	labelsPath = "coco.names"
	LABELS = open(labelsPath).read().strip().split("\n")

	np.random.seed(42)

	weightsPath = "./yolov3-tiny.weights"
	configPath = "./yolov3-tiny.cfg"

	net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	vs = cv2.VideoCapture(vidpath)
	writer = None
	(W, H) = (None, None)

	fl = 0
	q = 0
	while True:

	    (grabbed, frame) = vs.read()

	    if not grabbed:
	        break

	    if W is None or H is None:
	        (H, W) = frame.shape[:2]
	        q = W

	    frame = frame[0:H, 200:q]
	    (H, W) = frame.shape[:2]
	    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
	                                 swapRB=True, crop=False)
	    net.setInput(blob)
	    start = time.time()
	    layerOutputs = net.forward(ln)
	    end = time.time()

	    boxes = []
	    confidences = []
	    classIDs = []

	    for output in layerOutputs:

	        for detection in output:

	            scores = detection[5:]
	            classID = np.argmax(scores)
	            confidence = scores[classID]
	            if LABELS[classID] == "person":

	                if confidence > confid:
	                    box = detection[0:4] * np.array([W, H, W, H])
	                    (centerX, centerY, width, height) = box.astype("int")

	                    x = int(centerX - (width / 2))
	                    y = int(centerY - (height / 2))

	                    boxes.append([x, y, int(width), int(height)])
	                    confidences.append(float(confidence))
	                    classIDs.append(classID)

	    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confid, thresh)

	    if len(idxs) > 0:

	        status = list()
	        idf = idxs.flatten()
	        close_pair = list()#unsafe
	        s_close_pair = list()#safe
	        center = list()
	        dist = list()
	        for i in idf:
	            (x, y) = (boxes[i][0], boxes[i][1])
	            (w, h) = (boxes[i][2], boxes[i][3])
	            center.append([int(x + w / 2), int(y + h / 2)])

	            status.append(0)
	        for i in range(len(center)):
	            for j in range(len(center)):
	                g = isclose(center[i], center[j])

	                if g == 1:

	                    close_pair.append([center[i], center[j]])
	                    status[i] = 1
	                    status[j] = 1
	                elif g == 2:
	                    s_close_pair.append([center[i], center[j]])
	                    if status[i] != 1:
	                        status[i] = 2
	                    if status[j] != 1:
	                        status[j] = 2

	        total_p = len(center)
	        low_risk_p = status.count(2)
	        high_risk_p = status.count(1)
	        safe_p = status.count(0)
	        kk = 0

	        for i in idf:
	            (x, y) = (boxes[i][0], boxes[i][1])
	            (w, h) = (boxes[i][2], boxes[i][3])
	            if status[kk] == 1:
	                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 150), 2)

	            elif status[kk] == 0:
	                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

	            else:
	                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 120, 255), 2)

	            kk += 1
	        cv2.imshow('Social distancing analyser', frame)
	        cv2.waitKey(1)

	    if writer is None:
	        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
	        writer = cv2.VideoWriter("output.mp4", fourcc, 30,
	                                 (frame.shape[1], frame.shape[0]), True)

	    writer.write(frame)
	print("Processing finished: open output.mp4")
	writer.release()
	vs.release()

	print("flag %s" %flag)
	if(flag==-1):
		img(name)
	return 'success'

@app.route('/camera/<name>') 
def camera(name): 

	labelsPath = "./coco.names"
	LABELS = open(labelsPath).read().strip().split("\n")

	np.random.seed(42)
	COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
		dtype="uint8")

	weightsPath = "./yolov3-tiny.weights"
	configPath = "./yolov3-tiny.cfg"

	net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


	cap = cv2.VideoCapture(0)

	while(cap.isOpened()):
	    
	    ret,image=cap.read()
	    (H, W) = image.shape[:2]
	    ln = net.getLayerNames()
	    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
	    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
	    net.setInput(blob)
	    start = time.time()
	    layerOutputs = net.forward(ln)
	    end = time.time()
	    print("Frame Prediction Time : {:.6f} seconds".format(end - start))
	    boxes = []
	    confidences = []
	    classIDs = []
	    for output in layerOutputs:
	        for detection in output:
	            scores = detection[5:]
	            classID = np.argmax(scores)
	            confidence = scores[classID]
	            if confidence > 0.1 and classID == 0:
	                box = detection[0:4] * np.array([W, H, W, H])
	                (centerX, centerY, width, height) = box.astype("int")
	                x = int(centerX - (width / 2))
	                y = int(centerY - (height / 2))
	                boxes.append([x, y, int(width), int(height)])
	                confidences.append(float(confidence))
	                classIDs.append(classID)
	                
	    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.3)
	    ind = []
	    for i in range(0,len(classIDs)):
	        if(classIDs[i]==0):
	            ind.append(i)
	    a = []
	    b = []

	    if len(idxs) > 0:
	            for i in idxs.flatten():
	                (x, y) = (boxes[i][0], boxes[i][1])
	                (w, h) = (boxes[i][2], boxes[i][3])
	                a.append(x)
	                b.append(y)
	                
	    distance=[] 
	    nsd = []
	    for i in range(0,len(a)-1):
	        for k in range(1,len(a)):
	            if(k==i):
	                break
	            else:
	                x_dist = (a[k] - a[i])
	                y_dist = (b[k] - b[i])
	                d = math.sqrt(x_dist * x_dist + y_dist * y_dist)
	                distance.append(d)
	                if(d <=100):
	                    nsd.append(i)
	                    nsd.append(k)
	                nsd = list(dict.fromkeys(nsd))
	                print(nsd)
	    color = (0, 0, 255) 
	    for i in nsd:
	        (x, y) = (boxes[i][0], boxes[i][1])
	        (w, h) = (boxes[i][2], boxes[i][3])
	        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
	        text = "Alert"
	        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
	    color = (0, 255, 0) 
	    if len(idxs) > 0:
	        for i in idxs.flatten():
	            if (i in nsd):
	                break
	            else:
	                (x, y) = (boxes[i][0], boxes[i][1])
	                (w, h) = (boxes[i][2], boxes[i][3])
	                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
	                text = 'OK'
	                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)   
	    
	    cv2.imshow("Social Distancing Detector", image)
	    if cv2.waitKey(1) & 0xFF == ord('q'):
	        break
	cap.release()
	cv2.destroyAllWindows()

	print("flag %s" %flag)
	if(flag==-1):
		img(name)
	return 'success'

@app.route('/login',methods = ['POST', 'GET']) 
def login(): 
	if request.method == 'POST':
		file = request.files['file']
		if(file.filename==''):
			print('no image')
		else:	
			print(file)
			image(file)
			return redirect(url_for('login'))

		vid = request.files['vid']
		if(vid.filename==''):
			print('no vid')
		else:	
			print(vid)
			video(vid)
			return redirect(url_for('login'))

		cam= request.form['camera']
		if(cam=="Yes" or cam=="yes"):
			camera(cam)
		print(cam)	
		return redirect(url_for('login'))
	else: 
		return render_template('index_b.html')		
  
if __name__ == '__main__': 
   global flag
   flag=-1
   app.run(debug = True) 
