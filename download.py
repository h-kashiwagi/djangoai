import flickrapi
from urllib.request import urlretrieve #urlを取得する
import os,time,sys

flickr_api_key = "43439bc2beff6367431918d1030d3f3a"
secret_key = "5c243b30fca7e336"
wait_time = 1  #1秒間隔でリクエストを送信してデータを取得

keyword =sys.argv[1]
savedir = "./" + keyword

flickr =  flickrapi.FlickrAPI(flickr_api_key, secret_key, format='parsed-json')
#responseは辞書型の変数
response = flickr.photos.search( 
    #flicker.photos.searchのオプションを指定
    text = keyword,
    per_page = 400,
    media = 'photos',
    sort = 'relevance',
    safe_search = 1,
    extras = 'url_q, license'
)

photos = response['photos']  #photosデータ部分を取り出す


#データをダウンロード


for i, photo in enumerate(photos['photo']):   #photosのphotoに写真情報が格納されている
    url_q = photo['url_q']
    filepath = savedir + '/' + photo['id'] + '.jpg'
    if os.path.exists(filepath): continue
    urlretrieve(url_q,filepath)
    time.sleep(wait_time)