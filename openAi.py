import openai
from starlette.config import Config

config = Config(".env")
#OPEN_AI_KEY = config('OPENAI_API_KEY')



# 여기에 복사한 API 키를 입력하세요

def summarize_text_openai(text):
    new_text = '' 
    for i in text:
        new_text = new_text + ' ' + i
    # openai.api_key = OPEN_AI_KEY
    # response = openai.Completion.create(
    # model="text-davinci-003",
    # prompt=text,
    # temperature=0.7,
    # max_tokens=64,
    # top_p=1.0,
    # frequency_penalty=0.0,
    # presence_penalty=0.0
    # )
    # #print(response)
    # return response
    print('new_text: ', new_text)
    msg = openai.ChatCompletion.create(
                model = "gpt-3.5-turbo",
                messages = [
                    # GPT 역할 정의
                    {"role":"system", "content" : "너는 내가 보낸 문장을 요약해주는 역할을 해줄거야."},
                    # 원문
                    {"role":"user", "content" : new_text}
                    ],

                    # 답변의 창의적 범위 지정
                    temperature = 0.5,

                    # 답변에 가장 확률이 높은 범위 지정
                    top_p = 0.5,

                    # # 새로운 주제에 대한 샘플링 범위
                    presence_penalty = 1.5,

                    # 중복에 대한 범위
                    frequency_penalty = 1.5,

                    # 질문에 대한 답변 갯수
                    n = 1,

                    # 답변의 최대 토큰수 지정
                    max_tokens = 156,

                    # 중지 문자 설정(답변 끝까지 생성)
                    stop = None
    )
    return msg
    