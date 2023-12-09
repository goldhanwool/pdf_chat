import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import pytesseract
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import requests
import json
from starlette.config import Config
import time
import openai
from nltk.tokenize import sent_tokenize 
from papago import get_translate
from openAi import summarize_text_openai
import os
import nltk
from common_def import *
import re  

# Tesseract의 설치 경로를 설정합니다 (필요한 경우).
# 예: Windows에서 pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

name = 'test01'
image_path = './file/test.png'


folder_path = "./result/" + name         
pdf_folder_path = "./pdf/" + name

if not(os.path.isdir(folder_path)):  
    os.mkdir(folder_path) #new_folder create=> ./result/it’s Not Just Size That Matters 

# 이미지 불러오기
img = cv2.imread(image_path)
h, w, _ = img.shape

# 텍스트 영역 감지
t_info = pytesseract.image_to_data(img, output_type=Output.DICT) #output_type=Output.DICT: 추출된 데이터의 형식 -> 딕셔너리 타입

n_boxes = len(t_info['level'])
print("n_boxes: ", n_boxes)

from collections import Counter
v_list = t_info["height"]
most_common_overall_num, overall_count = Counter(v_list).most_common(1)[0]

mask = np.zeros((h, w), dtype=np.uint8)
for i in range(n_boxes):
    if int(t_info['conf'][i]) > 60:  # 확신도가 60 이상인 경우만 처리
        (x, y, w, h) = (t_info['left'][i], t_info['top'][i], t_info['width'][i], t_info['height'][i])
        if h < int(most_common_overall_num) + 10:
            # 텍스트 영역 추출
            mask[y:y+h, x:x+w] = 255
            
text_regions = cv2.bitwise_and(img, img, mask=mask)
cv2.imwrite('./output/text_regions.png', text_regions)

# 비텍스트 영역 추출
non_text_mask = cv2.bitwise_not(mask)
non_text_regions = cv2.bitwise_and(img, img, mask=non_text_mask)
cv2.imwrite('./output/non_text_regions.png', non_text_regions)

image = Image.open(image_path)
#텍스트 추출
#print('image: ', image)
text = pytesseract.image_to_string(image)
print('pytesseract-text: ', text)

words = []
words = text.splitlines( ) #print(words) # ['2003.02245v2 [cs.CL] 31 Jan 2021', '', 'arXiv', '', 'Data Augmentation Using Pre-trained Transformer Models', '', 'Varun Kumar', 'Alexa AI', 'kuvrun@amazon.com', '', 'Abstract', '', 'Language model based pre-trained models', 'such as BERT have provided significant gains', 'across different NLP tasks. In this paper, we', 'study different types of transformer based pre-', 'trained models such as auto-regressive models', '(GPT-2), auto-encoder models (BERT), and', 'seq2seq models (BART) for conditional data', 'augmentation. We show that prepending the', 'class labels to text sequences provides a simple', 'yet effective way to condition the pre-trained', 'models for data augmentation. Additionally,', 'on three classification benchmarks, pre-trained', 'Seq2Seq model outperforms other data aug-', 'mentation methods in a low-resource setting.', 'Further, we explore how different data aug-', 'mentation methods using pre-trained model', 'differ in-terms of data diversity, and how well', 'such methods preserve the class-label informa-', 'tion.', '', '1 Introduction', '', 'Data augmentation (DA) is a widely used technique', 'to increase the size of the training data. Increas-', 'ing training data size is often essential to reduce', 'overfitting and enhance the robustness of machine', 'learning models in low-data regime tasks.', '', 'In natural language processing (NLP), several', 'word replacement based methods have been ex-', 'plored for data augmentation. In particular, Wei', 'and Zou (2019) showed that simple word replace-', 'ment using knowledge bases like WordNet (Miller,', '1998) improves classification performance. Further,', 'Kobayashi (2018) utilized language models (LM)', 

new_list = []
openai_text = ''
cnt = 0
page_num = '{}'.format(i+1)
#파일편집하기
for i in words:
    if re.match('References', i) is not None or re.match('REFERENCES', i) is not None:
        break
    if len(new_list) == 0:
        new_list.append(i)
    elif i == '':
        cnt += 1
        new_list.append(i)
    # elif 'References' in i:
    #     print('References in i: ', i)
    #     break
    else:
        new_list[cnt] = new_list[cnt] + ' ' +i

# 페이지 요약 생성
openai_text = str(new_list)
summary_text = summary_openai(openai_text)
trans_text = translate_openai_summary(summary_text, name, page_num)   

# 파일저장하기
check_token_list = []
for i in new_list:
    sent_tokenize_list = sent_tokenize(i)
    for j in sent_tokenize_list:
        # 파파고 번역을 위해 특수문자 치환하기
        if '%' in j:
            j = j.replace('%', 'percent')
        check_token_list.append(j)
        #print('check_token_list: >>>>>>>>>>>>')
        #print(check_token_list)
        # 번역파일과 함께 저장하기
        print('j: ', j)
        trans_text = get_translate(j) #PaPaGo 번역
        
        f=open('./result/{}/1.{}_번역.txt'.format(name, name),'a',encoding='utf-8')
        #줄바꿈
        f.write('\n'+j+'\n'+trans_text+'\n')
        f.close()



#표 영역 추출
# import cv2
# import numpy as np

# def extract_non_table_areas(image_path):
#     img = cv2.imread(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)[1]

#     # 윤곽선 찾기
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # 전체 이미지 크기의 마스크를 생성
#     mask = np.ones(img.shape[:2], dtype="uint8") * 255

#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         # 마스크에서 표 영역을 검은색으로 채워 넣습니다
#         cv2.rectangle(mask, (x, y), (x+w, y+h), 0, -1)

#     # 마스크를 이용하여 표 영역을 제외한 부분만을 추출
#     non_table_regions = cv2.bitwise_and(img, img, mask=mask)

#     # 결과 이미지 저장
#     cv2.imwrite('non_table_regions.jpg', non_table_regions)

# # 이미지 경로
# image_path = 'page_0.jpg'
# extract_non_table_areas(image_path)
