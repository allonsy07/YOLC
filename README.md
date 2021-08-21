# YOLC
## CCP의 YOLC팀의 face recognition & blink detection 코드입니다.
  + 빵형의 개발도상국 코드를 기반으로 작성했습니다.
    + face recognition:
     ++ YOUTUBE : https://www.youtube.com/watch?v=3LNHxezPi1I&list=PL-xmlFOn6TULrmwkXjRCDAas0ixd_NtyK&index=61
     + CODE : https://github.com/kairess/simple_face_recognition
    + eye blink detection:
     + YOUTUBE : https://www.youtube.com/watch?v=dJjzTo8_x3c&list=PL-xmlFOn6TULrmwkXjRCDAas0ixd_NtyK&index=56
     + CODE :  https://github.com/kairess/eye_blink_detector

# Library
+ dlib
  + pip install dlib
  + or conda install dlib
  + or conda install -c conda-forge dlib
+ cv2
+ tensorflow(keras)
+ numpy
+ matplotlib

# Detail
+ 10번의 프레임동안 eye blink를 감지하고 저장합니다.
  + face detection 이후 eye crop을 진행한 뒤, 사전훈련된 model로 open, close를 classification합니다.
  + (classification model은 차후 개선이 필요할 것으로 판단됨)
+ eye blink가 5회 감지되었다면 face recognition으로 넘어갑니다.
  + eye blink가 감지되지 않고 특정 프레임이 지나면 사람이 아닌 사진으로 인식합니다.
  + face recognition은 사전에 사진이 등록된 사람에 한해 recognition이 가능합니다.
  + 사진을 등록하기 위해서는 다음과 같은 과정을 거쳐야 합니다.
   + img폴더에 사진을 업로드합니다.
   + face_registration.ipynb에서 img_paths와 descs를 업데이트합니다.
   + 코드를 실행시켜 사진을 embedding한 값을 descs.npy에 저장합니다.
+ 종료할 때엔 q를 누르면 됩니다.
