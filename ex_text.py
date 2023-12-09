import cv2
import numpy as np
import pytesseract
from pytesseract import Output

# Tesseract의 설치 경로를 설정합니다 (필요한 경우).
# 예: Windows에서 pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_and_non_text_areas(image_path):
    # 이미지 불러오기
    img = cv2.imread(image_path)
    h, w, _ = img.shape
    print(img.shape)
    print('h,w------------------')
    print(h, w)
    print('------------------')


    # 텍스트 영역 감지
    d = pytesseract.image_to_data(img, output_type=Output.DICT)
    n_boxes = len(d['level'])

    # 텍스트 영역을 표시할 마스크 생성
    mask = np.zeros((h, w), dtype=np.uint8)
    ls = [] 
    for i in range(n_boxes):
        if int(d['conf'][i]) > 60:  # 확신도가 60 이상인 경우만 처리
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            ls.append((y))
            # print('------------------')
            # print(x, y, w, h)
            # print("x","y","w","h")
            # print('------------------')
        
            mask[y:y+h, x:x+w] = 255

    # 텍스트 영역 추출
    text_regions = cv2.bitwise_and(img, img, mask=mask)
    cv2.imwrite('./output/text_regions.png', text_regions)

    # 비텍스트 영역 추출
    non_text_mask = cv2.bitwise_not(mask)
    non_text_regions = cv2.bitwise_and(img, img, mask=non_text_mask)
    cv2.imwrite('./output/non_text_regions.png', non_text_regions)

# 이미지 경로 지정
image_path = './file/test.png'
extract_text_and_non_text_areas(image_path)
