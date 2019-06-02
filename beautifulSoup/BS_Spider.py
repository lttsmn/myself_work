#encoding:UTF-8
from bs4 import BeautifulSoup
import requests
import time
import json
#BeautifulSoup 解析58同城网，里面主要用到BeautifulSoup 的select()方法
url = 'http://bj.58.com/pingbandiannao/24604629984324x.shtml'

wb_data = requests.get(url)
soup = BeautifulSoup(wb_data.text,'lxml')

#获取每件商品的URL
def get_links_from(who_sells):
    urls = []
    list_view = 'http://bj.58.com/pbdn/pn{}/'.format(str(who_sells))
    print ('list_view:{}'.format(list_view) )
    wb_data = requests.get(list_view)
    soup = BeautifulSoup(wb_data.text,'lxml')
    #for link in soup.select('td.t > a.t'):
    for link in soup.select('td.t  a.t'):  #跟上面的方法等价
        print (link)
        urls.append(link.get('href').split('?')[0])
    return urls


#获取58同城每一类商品的url  比如平板电脑  手机 等
def get_classify_url():
    url58 = 'http://bj.58.com'
    wb_data = requests.get(url58)
    soup = BeautifulSoup(wb_data.text, 'lxml')
    for link in soup.select('span.jumpBusiness a'):
        classify_href = link.get('href')
        print (classify_href)
        classify_url = url58 + classify_href
        print (classify_url)

#获取每件商品的具体信息
def get_item_info(who_sells=0):

    urls = get_links_from(who_sells)
    for url in urls:
        print (url)
        wb_data = requests.get(url)
        #print wb_data.text
        soup = BeautifulSoup(wb_data.text,'lxml')
        #print soup.select('infolist > div > table > tbody > tr.article-info > td.t > span.pricebiao > span')   ##infolist > div > table > tbody > tr.article-info > td.t > span.pricebiao > span
        print (soup.select('span[class="price_now"]')[0].text)
        print (soup.select('div[class="palce_li"]')[0].text)
        #print list(soup.select('.palce_li')[0].stripped_strings) if soup.find_all('div','palce_li') else None,  #body > div > div > div > div > div.info_massege.left > div.palce_li > span > i
        data = {
            'title':soup.title.text,
            'price': soup.select('span[class="price_now"]')[0].text,
            'area': soup.select('div[class="palce_li"]')[0].text if soup.find_all('div', 'palce_li') else None,
            'date' :soup.select('.look_time')[0].text,
            'cate' :'个人' if who_sells == 0 else '商家',
        }
        print(data)
        result = json.dumps(data, encoding='UTF-8', ensure_ascii=False) #中文内容仍然无法正常显示。 使用json进行格式转换，然后打印输出。
        print (result)

# get_item_info(url)

# get_links_from(1)

get_item_info(1)
#get_classify_url()