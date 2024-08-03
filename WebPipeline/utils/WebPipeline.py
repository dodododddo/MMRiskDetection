import requests
from bs4 import BeautifulSoup
import re 
from urllib.parse import urljoin
from .remove_whitespace import remove_whitespace
import random
from requests import Response

class WebData:
    def __init__(self, text:str, image_paths:list):
        self.text = text
        self.image_paths = image_paths

    def __str__(self):
        return str(self.__dict__)


class WebPipeline():
    def __init__(self, headers=[
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0"
    "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.153 Safari/537.36",
    "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:30.0) Gecko/20100101 Firefox/30.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.75.14 (KHTML, like Gecko) Version/7.0.3 Safari/537.75.14",
    "Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.2; Win64; x64; Trident/6.0)",
    'Mozilla/5.0 (Windows; U; Windows NT 5.1; it; rv:1.8.1.11) Gecko/20071127 Firefox/2.0.0.11',
    'Opera/9.25 (Windows NT 5.1; U; en)',
    'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)',
    'Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.5 (like Gecko) (Kubuntu)',
    'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.8.0.12) Gecko/20070731 Ubuntu/dapper-security Firefox/1.5.0.12',
    'Lynx/2.8.5rel.1 libwww-FM/2.14 SSL-MM/1.4.1 GNUTLS/1.2.9',
    "Mozilla/5.0 (X11; Linux i686) AppleWebKit/535.7 (KHTML, like Gecko) Ubuntu/11.04 Chromium/16.0.912.77 Chrome/16.0.912.77 Safari/535.7",
    "Mozilla/5.0 (X11; Ubuntu; Linux i686; rv:10.0) Gecko/20100101 Firefox/10.0 "], 
    cookie = 'X_CACHE_KEY=3b823947ad6a3f3706f3f67517203722; mx_style=white; _pk_id.1.b4fd=8dbc9a4f6013e1b7.1720616245.; _pk_ses.1.b4fd=1; cf_clearance=mFAnZFNDi2il0gqaF3O_0ExyA3Lpf8mWrA6UGf_jCR8-1720616245-1.0.1.1-Lh7nUQMyd12V5ngD3JHFcvD4DGBHDjVwpUE8xIXPvkDblqWVJltgZLRMymJujJRk3.vrR79gr6aOLUQfsXZ15A; showBtn=true'):
        
        self.headers = headers
        self.cookie = cookie

    def _get_response(self, url):
        headers = {'User-Agent': random.choice(self.headers), 'Cookie': self.cookie, 'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8'}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response
        else:
            print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
            return False
            
    def _get_soup(self, response:Response):
         soup = BeautifulSoup(response.content, 'html.parser')
         return soup

    def _get_text(self, soup):
        return remove_whitespace(soup.get_text())
    
    def _get_image(self, url, soup):
        image_links = soup.find_all('img')
        image_paths = [urljoin(url, image_link.get('src')) 
                       if not image_link.get('src', '').startswith(('http://', 'https://')) 
                       else image_link.get('src') 
                       for image_link in image_links]
        return image_paths 

    def __call__(self, url) -> WebData:
        response = self._get_response(url)
        soup = self._get_soup(response)
        text = self._get_text(soup)
        image_paths = self._get_image(url, soup)
        return WebData(text, image_paths)