import cv2, dlib
import numpy as np
from imutils import face_utils
from keras.models import load_model

IMG_SIZE = (34, 26)
blink_detection = True
face_recognition = False

detector = dlib.get_frontal_face_detector() # face detection
predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat') # face shape predictor : to find out location of eyes
facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat') #face recognition, embedding face

descs = np.load('img/descs.npy')[()] # embedded face images

model = load_model('models/2018_12_17_22_58_35.h5') # predict whether the eyes are open

def crop_eye(img, eye_points):
  x1, y1 = np.amin(eye_points, axis=0)
  x2, y2 = np.amax(eye_points, axis=0)
  cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

  w = (x2 - x1) * 1.2
  h = w * IMG_SIZE[1] / IMG_SIZE[0]

  margin_x, margin_y = w / 2, h / 2

  min_x, min_y = int(cx - margin_x), int(cy - margin_y)
  max_x, max_y = int(cx + margin_x), int(cy + margin_y)

  eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(int)

  eye_img = gray[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

  return eye_img, eye_rect

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

prev_l, prev_r = [], []
steps, blink_l, blink_r, blink_count = 0, 0, 0, 0

while cap.isOpened():
  ret, img_ori = cap.read()

  if not ret:
    break

  img_ori = cv2.resize(img_ori, dsize=(0, 0), fx=0.5, fy=0.5)
  img = img_ori.copy()

  if blink_detection:    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    steps += 1

    for face in faces:
      shapes = predictor(gray, face) # detect face from camera
      shapes = face_utils.shape_to_np(shapes) # change form into numpy

      eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42]) # find left eyes and crop
      eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48]) # find rigth eyes and crop

      eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE) # resize eyes
      eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE) # resize eyes
      eye_img_r = cv2.flip(eye_img_r, flipCode=1) # flip right eyes, so it can fit in model to predict whether the eyes are open

      eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.
      eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.

      pred_l = model.predict(eye_input_l)
      pred_r = model.predict(eye_input_r)

      state_l = 'O' if pred_l > 0.1 else 'X'
      state_r = 'O' if pred_r > 0.1 else 'X'

      if len(prev_l) < 10:
          prev_l.append(state_l)
          prev_r.append(state_r)
          steps -= 1
      else: 
          for count in range(9):
                if prev_l[count] != prev_l[count + 1]:
                      blink_l += 1
                if prev_r[count] != prev_r[count + 1]:
                      blink_r += 1
          if blink_l > 1 and blink_r > 1:
              print('Eye blink is estimated!')
              prev_l, prev_r = ['O'], ['O']
              blink_l, blink_r = 0, 0
              steps = 0
              blink_count += 1
              if blink_count == 5:
                print('Eye blink is estimated five times! You are human')
                print('Now I will recognize your face')
                blink_detection = False
                face_recognition = True
          elif steps == 20:
            print('Eye blink isn\'t estimated! Please blink your eyes')
          elif steps == 40:
            print('Eye blink isn\'t estimated! You are Image, I think!') 
            steps = 0     
            blink_count = 0

          prev_l.pop(0)
          prev_r.pop(0)

      # visualize
      cv2.rectangle(img, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(255,255,255), thickness=2)
      cv2.rectangle(img, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(255,255,255), thickness=2)

      cv2.putText(img, state_l, tuple(eye_rect_l[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
      cv2.putText(img, state_r, tuple(eye_rect_r[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

  elif face_recognition:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faces = detector(img, 1)

    for k, d in enumerate(faces):
      shape = predictor(img_rgb, d)
      face_descriptor = facerec.compute_face_descriptor(img_rgb, shape)

      last_found = {'name': 'unknown', 'dist': 0.6, 'color': (0,0,255)}

      for name, saved_desc in descs.items():
        dist = np.linalg.norm([face_descriptor] - saved_desc, axis=1)

        if dist < last_found['dist']:
          last_found = {'name': name, 'dist': dist, 'color': (255,255,255)}

      cv2.rectangle(img, pt1=(d.left(), d.top()), pt2=(d.right(), d.bottom()), color=last_found['color'], thickness=2)
      cv2.putText(img, last_found['name'], org=(d.left(), d.top()), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=last_found['color'], thickness=2)
    
  cv2.imshow('img', img)
  if cv2.waitKey(1) == ord('q'):
    break

  
