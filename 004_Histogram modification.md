---
# 4.Histogram modification
---
## 4.1. Histgram stretching


```python
import sys
import numpy as np
import cv2

src = cv2.imread('fig/Hawkes.jpg', cv2.IMREAD_GRAYSCALE)

if src is None:
    print('Image load failed!')
    sys.exit()
    
# minMaxLoc(src[, mask]) -> minVal, maxVal, minLoc, maxLoc
smin, smax, _, _ = cv2.minMaxLoc(src)
print(smin, smax)
# cv2.normalize(src, dst=None, alpha=None, beta=None, norm_type=None, dtype=None, mask=None) -> dst
# src: 입력영상
# dst: 결과영상
# alpha: 정규화 최소값 (예, 0)
# beta: 정규화 최댓값 (예, 155)
# norm_type: cv2.NORM_MINMAX
# dtype =  -1, src와 같은 type

# dst = cv2.normalize(src, None, 0, 255, cv2.NORM_MINMAX, -1)
dst = np.clip(255*(src-smin)/(smax-smin) + 0, 0, 255).astype(np.uint8)

# dst = (255*(src-smin)/(smax-smin) + 0)
# dst = dst.astype(np.uint8)

dmin, dmax, _, _ = cv2.minMaxLoc(dst)
print(dmin, dmax)

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()

cv2.destroyAllWindows()
```

    113.0 213.0
    0.0 255.0
    

## 4.2. 히스토그램 평활화


```python
import sys
import numpy as np
import cv2


# 그레이스케일 영상의 히스토그램 평활화
src = cv2.imread('fig/Hawkes.jpg', cv2.IMREAD_GRAYSCALE)

if src is None:
    print('Image load failed!')
    sys.exit()

# equalizeHist(src, dst=None) -> dst
# src: 입력영상,gray scale 영상만 가능
dst = cv2.equalizeHist(src)

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()

cv2.destroyAllWindows()

# 컬러 영상의 히스토그램 평활화
src = cv2.imread('fig/field.bmp')

if src is None:
    print('Image load failed!')
    sys.exit()

src_ycrcb = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)
ycrcb_split = cv2.split(src_ycrcb) # list ret

# 밝기 성분에 대해서만 히스토그램 평활화 수행
ycrcb_split[0] = cv2.equalizeHist(ycrcb_split[0])

# dst_ycrcb = cv2.merge(ycrcb_split)
dst = cv2.cvtColor(dst_ycrcb, cv2.COLOR_YCrCb2BGR)

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()

cv2.destroyAllWindows()

## 히스토그램 평활화를 통해서 개선할 수 있는 영상을 찾고 평활화 수행
```

## 4.3. 특정 색상 영역 찾아 내기


```python
import sys
import numpy as np
import cv2


src = cv2.imread('fig/candies.png')
# src = cv2.imread('fig/candies2.png')

if src is None:
    print('Image load failed!')
    sys.exit()

src_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

# inRange(src, lowerb, upperb[, dst]) -> dst
# src: 입력영상
# lowerb: 하한값
# upperb: 상한값
dst1 = cv2.inRange(src, (0, 128, 0), (100, 255, 100)) # b, g, r
dst2 = cv2.inRange(src_hsv, (50, 150, 0), (80, 255, 255)) # h, s, v

cv2.imshow('src', src)
cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)

cv2.waitKey()

cv2.destroyAllWindows()
```
