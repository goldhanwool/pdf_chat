from starlette.config import Config
import json
import urllib.request

config = Config(".env")
PAPAGO_CLIENT = config('PAPAGO_CLIENT')
PAPAGO_SECRET = config('PAPAGO_SECRET')

def get_translate(text):
    try:
        if text is None:
            return 0
        client_id = PAPAGO_CLIENT
        client_secret = PAPAGO_SECRET
        encText = text

        ''' 한국어 -> 영어: source=ko&target=en&text=
            영어 -> 한국어: source=en&target=ko&text=
        '''
        data = "source=en&target=ko&text=" + encText
        #data = "source=ko&target=en&text=" + encText
        url = "https://naveropenapi.apigw.ntruss.com/nmt/v1/translation"
        request = urllib.request.Request(url)
        request.add_header("X-NCP-APIGW-API-KEY-ID",client_id)
        request.add_header("X-NCP-APIGW-API-KEY",client_secret)
        response = urllib.request.urlopen(request, data=data.encode("utf-8"))
        rescode = response.getcode()
        if(rescode==200):
            response_body = response.read()
            #print(response_body.decode('utf-8'))
            res = json.loads(response_body.decode('utf-8'))
            #print('res: ' , res['message']['result']['translatedText'])
            return res['message']['result']['translatedText']
        else:
            print("Error Code:" + rescode)
    except Exception as e:
        print(e)
        return '*******번역실패: 요청텍스트 => {}, error => {}****************'.format(text, e)

