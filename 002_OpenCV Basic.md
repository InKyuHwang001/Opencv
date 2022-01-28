---
# 2. OpenCV Basic
---
## 2.1. 영상속성


```python
import sys
import cv2
import numpy as np
```

## 2.2. 영상 불러오기


```python
img1 = cv2.imread('fig/puppy.bmp', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('fig/puppy_1280_853.jpg', cv2.IMREAD_COLOR)

if img1 is None or img2 is None:
    print('Image load failed!')
    sys.exit()


# 영상의 속성 참조
print('type(img1):', type(img1))
print('img1.shape:', img1.shape)
print('img2.shape:', img2.shape)
print('img1.dtype:', img1.dtype)
print('img2.dtype:', img2.dtype)

print('img1.shape length:', len(img1.shape))
print('img2.shape length:', len(img2.shape))
```

    type(img1): <class 'numpy.ndarray'>
    img1.shape: (480, 640)
    img2.shape: (853, 1280, 3)
    img1.dtype: uint8
    img2.dtype: uint8
    img1.shape length: 2
    img2.shape length: 3
    

## 2.3. 영상의 크기 참조


```python
h, w = img1.shape
print('img1 size: {} x {}'.format(w, h))

h, w = img2.shape[:2]
print('img2 size: {} x {}'.format(w, h))
```

    img1 size: 640 x 480
    img2 size: 1280 x 853
    

## 2.4. 영상의 픽셀값 참조


```python
x = 230
y = 320

p1 =img1[y,x]
print(p1)

p2 = img2[y, x]
print(p2)

### 픽셀값 바꾸기
img1[10:200, 10:200] = 0
img2[10:200, 10:200] = (0, 0, 255)

cv2.imshow('image', img1)
cv2.imshow('image2',img2)

cv2.waitKey()
cv2.destroyAllWindows()
```

    128
    [210 216 227]
    

## 2.5.영상생성


```python
import numpy as np
import cv2

# 새 영상 생성하기

img1 = np.random.randint(0, 255, (240, 320), dtype = np.uint8) # gray random scale 
img2 = np.zeros((240, 320, 3), dtype=np.uint8)    # color image
img3 = np.ones((240, 320), dtype=np.uint8) * 255  # dark gray
img4 = np.full((240, 320, 3), (0, 255, 255), dtype=np.uint8)  # yellow


cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.imshow('img3', img3)
cv2.imshow('img4', img4)


cv2.waitKey()
cv2.destroyAllWindows()
```

## 2.6. 영상 복사


```python
# 영상 복사
img1 = cv2.imread('fig/puppy.bmp', cv2.IMREAD_COLOR)
# img1 = cv2.imread('HappyFish.jpg')

if img1 is None:
    print("image load failed")
    sys.exit()

img2 = img1
# img2 = img1[150:250, 200:500]

img3 = img1.copy()

# img2[:] = (0, 0, 255)

img1[200:300,240:400] = (0, 255, 255)

print(img1.shape)
print(img1.dtype)

cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.imshow('img3', img3)

cv2.waitKey()
cv2.destroyAllWindows()

```

    (480, 640, 3)
    uint8
    

## 2.7. 부분 영상 추출


```python
img1 = cv2.imread('fig/puppy.bmp')

img2 = img1[200:400, 300:500]  # numpy.ndarray의 슬라이싱
img3 = img1[200:400, 300:500].copy()

# img1.fill(255)
# circle(img, center, radius, color[, thickness[, lineType[, shift]]]) -> img
cv2.circle(img2, (100, 100), 50, (0, 0, 255), 3)

cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.imshow('img3', img3)

cv2.waitKey()
cv2.destroyAllWindows()
```

## 2.8. 마스크 연산과 ROI


```python
# 마스크 영상을 이용한 영상 합성 
# cv2.copyTo(src, mask, dst = None) -> dst

src = cv2.imread('fig/airplane.bmp', cv2.IMREAD_COLOR)
mask = cv2.imread('fig/mask_plane.bmp', cv2.IMREAD_GRAYSCALE)
dst = cv2.imread('fig/field.bmp', cv2.IMREAD_COLOR)

if src is None or mask is None or dst is None:
    print('Image read failed!')
    sys.exit()
    

# 영상의 포맷과 형식이 같아야 함
cv2.copyTo(src, mask, dst)
# dst = cv2.copyTo(src, mask)

# Using numpy
# dst[mask > 0] = src[mask > 0]

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.imshow('mask', mask)

cv2.waitKey()
cv2.destroyAllWindows()
```

## 2.9.송아지 합성


```python
img1 = cv2.imread('fig/cow.png')
img2 = cv2.imread('fig/green.png')
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

print(img1.shape)
print(img2.shape)

h, w = img1.shape[:2]

img2 = cv2.resize(img2, (w, h), cv2.INTER_AREA)
ret, mask = cv2.threshold(img1_gray, 244, 255, cv2.THRESH_BINARY_INV)

cv2.copyTo(img1, mask, img2)

cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.imshow('img1_gray', img1_gray)
cv2.imshow('mask', mask)


cv2.waitKey()
cv2.destroyAllWindows()
```

    (600, 600, 3)
    (1125, 1500, 3)


## 2.10. 알파 채널을 마스크 영상으로 이용


```python
src = cv2.imread('fig/puppy.bmp', cv2.IMREAD_COLOR)
sunglass = cv2.imread('fig/imgbin_sunglasses_1.png', cv2.IMREAD_UNCHANGED)

sunglass = cv2.resize(sunglass, (300, 150))

if src is None or sunglass is None:
    print('Image read failed!')
    sys.exit()


mask = sunglass[:, :, -1]    # mask는 알파 채널로 만든 마스크 영상
glass = sunglass[:, :, 0:3]  # glass는 b, g, r 3채널로 구성된 컬러 영상

h, w = mask.shape[:2]
crop = src[120:120+h, 220:220+w]  # glass mask와 같은 크기의 부분 영상 추출

cv2.copyTo(glass, mask, crop) #<1> 안경을 겁개 쓸때
#crop[mask > 0] = (0, 0, 255) #<2> 안경에 색을 입힐때 <1>,<2>둘중 하나만 활성화

cv2.imshow('src', src)
cv2.imshow('glass', glass)
cv2.imshow('mask', mask)
cv2.imshow('crop', crop)

# cv2.imwrite('puppy_sunglass.bmp', src)

cv2.waitKey()
cv2.destroyAllWindows()
```

## 2.11.OpenCV 그리기 함수


```python
img=np.full((600,1200,3),255, np.uint8)

# cv2.line(img, pt1, pt2, color, thickness = None, lineType = None, shift = None) -> img
# flags
    # img:그림을 그릴 영상
    # pt1, pt2: 직선의 시작점, 끝점(영상좌표)
    # color: 직선의 칼라 (B,G,R)의 튜플
    # thinkness: 선두께, 기본은= 1
    # lineType: cv2.LINE_4, cv2.LINE_8, cv2.LINE_AA
cv2.line(img, (100,50),(300,50),(0,0,255),10)
cv2.line(img, (300,50),(150,150),(0,0,255),10)

cv2.arrowedLine(img, (400,50),(400,250),(0,0,255),10)

# cv2.rectangle(img, pt1, pt2, color, thickness = None, lineType = None) -> img
   # pt1 :좌측 상단,  pt2: 우측하단
# cv2.rectangle(img, rect, color, thickness = None, lineType = None) -> img
    # rect: 사각형의 위치 정보 (x, y, w, h)
cv2.rectangle(img, (100,300),(400,400),(255,0,255),-1)
cv2.rectangle(img, (100,300,300,100),(0,0,255),10)

# cv2.circle(img, center, radius, color, thickness = None, lineType = None) -> img
    # center: 원의 중심좌표 (x, y)
    # radius : 원의 반지름
cv2.circle(img,(600,300),100,(255,0,255),10,cv2.LINE_AA)

# cv2.ellips(img, center, axes, angle, startAngle, endAngle, color[, thickness[, lineType[, shift]]]) -> img
    # center: 원의 중심좌표 (x, y)
    # axis: 축의 반지름(x, y)
    # angle: 타원의 기울기 (예, 10, 오른쪽으로 10도 기울어짐)
    # startAngle: 타원을 그리는 시작 각도 (3시 방향이 0도)
    # endAngle: 타원을 그리는 종료 각도
cv2.ellipse(img,(600,300),(50,100),10,0,360,(0,255,0),10)

# cv2.polylines(img, pts, isClosed, color, thickness = None, lineType = None) -> img
    # center: 다각형 점들의 array
    # isClosed : True for 폐곡선

text='Opencv version= '+cv2.__version__
cv2.putText(img, text,(800,100),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),1,cv2.LINE_AA)

print(text)
cv2.imshow('canvas', img)

cv2.waitKey()
cv2.destroyAllWindows()
```

    Opencv version= 4.5.5


## 2.12. 카메라와 동영상 처리하기 


```python
cap =cv2.VideoCapture(0)

if not cap.isOpened():
    print('Videocap open failed')
    sys.exit()

# 카메라 프레임 처리
while True:
    ret, frame=cap.read()
    
    if not ret:
        print('video read failed')
        break
    edge = cv2.Canny(frame, 50, 150) 
    inversed = ~edge  # 반전  
    
    cv2.imshow('img',frame)
    cv2.imshow('frame1', edge)
    cv2.imshow('inversed', inversed)
    if cv2.waitKey(20)==27:
        break
        
cap.release()
cv2.destroyAllWindows()
```


```python
cap =cv2.VideoCapture(0)

if not cap.isOpened():
    print('Videocap open failed')
    sys.exit()

w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps=int(cap.get(cv2.CAP_PROP_FPS)*0.7) 
fourcc=cv2.VideoWriter_fourcc(*'DIVX') # *'DIVX' == 'D', 'I', 'V', 'X'

out = cv2.VideoWriter('output.avi', fourcc, fps, (w, h))

# 카메라 프레임 처리
while True:
    ret, frame=cap.read()
    
    if not ret:
        print('video read failed')
        break
    ##동영상 편십부분
    ###############################
    edge = cv2.Canny(frame, 50, 150) 
    edge_color=cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
    ###############################
    cv2.imshow('edge', edge_color)
    
    out.write(edge)
    
    if cv2.waitKey(20)==27:
        break
        
cap.release()
out.release()
cv2.destroyAllWindows()
```


```python
img = cv2.imread('./fig/cat.bmp', cv2.IMREAD_GRAYSCALE)

if img is None:
    print('Image load failed!')
    sys.exit()

cv2.namedWindow('image')
cv2.imshow('image', img)

# cv2.waitKeyEx()

while True:
    
    keycode = cv2.waitKey()
    
    if keycode == ord('i'):
        img = ~img        
        cv2.imshow('image', img)
    
    elif keycode == ord('e'):
        img = cv2.Canny(img, 50, 150)
        
        cv2.imshow('image', img)
    
    elif keycode == 27:
        break

cv2.destroyAllWindows()
```

## 2.13. 마우스 이벤트 처리하기


```python
oldx = oldy = 0

def call_mouse(event, x, y, flags, param):
    global oldx, oldy

    if event == cv2.EVENT_LBUTTONDOWN:
        oldx, oldy = x, y
        print('EVENT_LBUTTONDOWN: %d, %d' % (x, y))

    elif event == cv2.EVENT_LBUTTONUP:
        print('EVENT_LBUTTONUP: %d, %d' % (x, y))

    elif event == cv2.EVENT_MOUSEMOVE:
        if flags & cv2.EVENT_FLAG_LBUTTON:
            cv2.line(img, (oldx, oldy), (x, y), (0, 0, 255), 4, cv2.LINE_AA)
            cv2.imshow('image', img)
            oldx, oldy = x, y


img = np.ones((480, 640, 3), dtype=np.uint8) * 255

cv2.namedWindow('image')

# cv2.setMouseCallback(windowName, onMouse, param = None) -> None
    # windowName: 마우스이벤트를 수행할 창 이름
    # onMouse: 마우스 이벤트 콜벡함수
    # param: 콜백함수에 전달할 데이터

# onMouse(event, x, y, flags, param) -> None
# event: 마우스 이벤트 종류 e.g., cv2.EVENT_LBUTTONDOWN
# x, y : 창을 기준으로 이벤트 발생좌표
# flags: 이벤트시 발생 상태 e.g., "ctrl"
# param: cv2.setMouseCallback()함수에서 설정한 데이터

# 마우스 event와 flags reference
# docs/opencv.org/master -> MouseEventTypes cv, MouseEventFlags cv

cv2.setMouseCallback('image',call_mouse, img)
cv2.imshow('image', img)

cv2.waitKey()
cv2.destroyAllWindows()
```

    EVENT_LBUTTONDOWN: 175, 112
    EVENT_LBUTTONUP: 380, 154


## 2.14. 트랙바 사용하기


```python
def call_trackbar(pos):
    print(pos)
    value = 256
#     if value >= 255:
#         value = 255
#     value = np.clip(value,0,255)

    img[:] = pos
    cv2.imshow('image', img)

img = np.zeros((480, 640), np.uint8)
cv2.namedWindow('image')

# createTrackbar(trackbarName, windowName, value, count, onChange) -> None
# trackbarName: 트랙바 이름
# windowName : 트랙바를 생성할 창 이름
# value : 트랙바 위치 초기값
# count : 트랙바 최댓값, 최솟값은 0
# onChange :callback 함수 e.g., onChange(pos) 위치를 정수형태로 전달
cv2.createTrackbar('level', 'image', 0, 256, call_trackbar) # 창이 생성된 후 호출

cv2.imshow('image', img)
cv2.waitKey()
cv2.destroyAllWindows()
```


```python
def call_track(pos):
    global img
    
    img_glass=img*pos
    cv2.imshow('image',img_glass)

img_alpha=cv2.imread('./fig/imgbin_sunglasses_1.png',cv2.IMREAD_UNCHANGED)
img=img_alpha[:,:,-1]

img[img>0]=1

cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.imshow('image',img)
cv2.createTrackbar('level','image',0,255,call_track)

cv2.waitKey()
cv2.destroyAllWindows()
```


