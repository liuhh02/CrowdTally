import streamlit as st
from iterdet.inference import inference_detector, init_detector
import numpy as np
import requests
from PIL import Image
import mmcv
import tempfile
import uuid
import pyrebase
import imutils
import cv2
import time
from config import firebaseConfig

MODEL_PATH = './yolo_model'
labelsPath = os.path.sep.join([MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# NOTE: the yolov4.weights file is too large to be uploaded on GitHub,
#       the file can be found at https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
weightsPath = os.path.sep.join([MODEL_PATH, "yolov4.weights"])
configPath = os.path.sep.join([MODEL_PATH, "yolov4.cfg"])

# firebase config
firebase = pyrebase.initialize_app(firebaseConfig)
storage = firebase.storage()

@st.cache(allow_output_mutation=True)
def loadmodel():
  config = 'config.py'
  checkpoint = './model.pth'
  model = init_detector(config, checkpoint, device='cuda:0')
  return model

def detect_people(frame, net, ln, MIN_CONF, personIdx=0):
	(H, W) = frame.shape[:2]
	results = []
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	layerOutputs = net.forward(ln)

	boxes = []
	centroids = []
	confidences = []

	for output in layerOutputs:
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			if classID == personIdx and confidence > MIN_CONF:
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				boxes.append([x, y, int(width), int(height)])
				centroids.append((centerX, centerY))
				confidences.append(float(confidence))

	idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, 0.3)

	if len(idxs) > 0:
		for i in idxs.flatten():
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			r = (confidences[i], (x, y, x + w, y + h), centroids[i])
			results.append(r)
	return results

def yolo(video, MIN_CONF):
	print("Initializing YOLO...")
	net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	print("Analyzing video")
	vf = cv2.VideoCapture(video)

	stframe = st.empty()
	while vf.isOpened():
		(grabbed, frame) = vf.read()
		if not grabbed:
			break

		frame = imutils.resize(frame, width=700)
		results = detect_people(frame, net, ln, MIN_CONF, personIdx=LABELS.index("person"))

		for (i, (prob, bbox, centroid)) in enumerate(results):
			(startX, startY, endX, endY) = bbox
			(cX, cY) = centroid
			color = (103, 214, 133)

			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
			cv2.circle(frame, (cX, cY), 5, color, 1)

		text = f"{len(results)} People Detected"
		cv2.putText(frame, text, (10, 25),
			cv2.FONT_HERSHEY_SIMPLEX, 0.75, (26, 106, 135), 3)

		stframe.image(frame)

st.set_page_config(page_title="CrowdTally", page_icon="🌊", layout="wide")

def main():
  st.title('Count the Number of Beachgoers')
  model = loadmodel()
  
  functionality = st.sidebar.selectbox('What would you like to do?', ('Count from Image', 'Count from Live Video'))
  
  if (functionality == 'Count from Image'):
    st.subheader('Count from Image')
    choice = st.sidebar.selectbox('Choose your image', ('Upload my own image', 'Search for crowdsourced images'))
	
    if choice == 'Upload my own image':
      f = st.file_uploader("Upload an image")
      if f is not None:
        image = Image.open(f)
        img = np.asarray(image)
        st.write("**Uploaded Image**")
        st.image(img, width=700)
        result = inference_detector(model, img)
        num = 0
        st.sidebar.markdown("# Configure the Model")
        confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.3, 0.05)
        for box in result[0]:
          score = box[-1]
          if score > confidence_threshold:
            num += 1
        st.subheader(f"{num} people detected at the beach")
        st.write("Adjust the confidence threshold on the sidebar accordingly! If the model is detecting too many people, increase the confidence threshold. If too few people are detected, decrease the threshold.")
        result_img = model.show_result(img, result, score_thr=confidence_threshold, show=False)
        st.write("**Annotated Image**")
        st.image(result_img, width=700)
        upload = st.button("Would you like to upload this image for others to view?")
        if upload:
          filename = uuid.uuid4()
          storage.child(f"{filename}.jfif").put(f)
          st.write("Success! Your image has been uploaded. Thank you for contributing to our crowdsourced database of images!")
	
    else:
      files = storage.list_files()
      url_list = []
      for file in files:
        url_list.append(storage.child(file.name).get_url(None))
      col1, col2 = st.beta_columns(2)
      with col1:
        st.image(url_list[0])
        choice1 = st.button("Select this image", key=1)
        st.image(url_list[1])
        choice2 = st.button("Select this image", key=2)
        st.image(url_list[2])
        choice3 = st.button("Select this image", key=3)
      with col2:
        st.image(url_list[3])
        choice4 = st.button("Select this image", key=4)
        st.image(url_list[4])
        choice5 = st.button("Select this image", key=5)
        st.image(url_list[5])
        choice6 = st.button("Select this image", key=6)
      link = '[Search for more images on Unsplash](https://unsplash.com/s/photos/crowd-at-beach)'
      st.markdown(link, unsafe_allow_html=True)
      if choice1:
        image = Image.open(requests.get(url_list[0], stream=True).raw)
      elif choice2:
        image = Image.open(requests.get(url_list[1], stream=True).raw)
      elif choice3:
        image = Image.open(requests.get(url_list[2], stream=True).raw)
      elif choice4:
        image = Image.open(requests.get(url_list[3], stream=True).raw)
      elif choice5:
        image = Image.open(requests.get(url_list[4], stream=True).raw)
      elif choice6:
        image = Image.open(requests.get(url_list[5], stream=True).raw)
      st.sidebar.markdown("# Configure the Model")
      confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.3, 0.05)
      if choice1 or choice2 or choice3 or choice4 or choice5 or choice6:
        img = np.asarray(image)
        result = inference_detector(model, img)
        num = 0
        for box in result[0]:
          score = box[-1]
          if score > confidence_threshold:
            num += 1
        st.subheader(f"{num} people detected at the beach")
        st.write("Adjust the confidence threshold on the sidebar accordingly! If the model is detecting too many people, increase the confidence threshold. If too few people are detected, decrease the threshold.")
        result_img = model.show_result(img, result, score_thr=confidence_threshold, show=False)
        st.write("**Annotated Image**")
        st.image(result_img, width=700)
	
  elif functionality == 'Count from Live Video':
    model_type = st.sidebar.selectbox("Choose your model type", ("Best Precision (IterDet)", "Lightweight (YOLO)"))

    if model_type == "Best Precision (IterDet)":
      f = st.file_uploader("Upload a video")
      if f is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(f.read())
        video = mmcv.VideoReader(tfile.name)
        stframe = st.empty()
        st.sidebar.markdown("# Configure the Model")
        confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.3, 0.05)
        for frame in video:
          result = inference_detector(model, frame)
          num = 0
          for box in result[0]:
            score = box[-1]
            if score > confidence_threshold:
              num += 1
          new = model.show_result(frame, result)
          text = f"{num} People Detected"
          cv2.putText(new, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (135, 106, 26), 3)
          stframe.image(new, channels="BGR")
        stframe.empty()
	
      elif model_type == "Lightweight (YOLO)":
        f = st.file_uploader("Upload a video")
        if f is not None:
          while True:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(f.read())
            yolo(tfile.name, confidence_threshold, overlap_threshold)

if __name__=='__main__':
  main()
