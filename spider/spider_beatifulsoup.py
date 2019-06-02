from bs4 import BeautifulSoup#导入模块
import requests #网页访问库
res=requests.get("https://home.firefoxchina.cn/")
res.encoding="utf-8"

be=BeautifulSoup(res.text,"html")#得到BeautifulSoup对象，lxml为HTML解析器，如XML解析则要用xml
print(be.original_encoding)#输出编码
print(be.prettify())#以标准HTML格式输出网页源码

print(be.input)#获取到第一个input标签全部内容：<input name="img" type="file"/>
print(be.form.input)#获取到标签（form）下的子标签(input)
print(be.form.encode("latin-1"))#自定义编码输出
print(be.input.parent.parent)#获取input标签的父节点的父节点
print(be.input.previous_sibling)#上一个兄弟节点
print(be.input.next_sibling)#下一个兄弟节点
print(be.img)#获取到第一个img标签内容：<img src="img/0.jpg"/>
picture=be.img
print(picture.get('src'))#获取该属性值（优先考虑）：img/0.jpg
print(be.img["src"])#直接获取属性值

#获取到标签内容值
print(be.title) # <title>东小东页</title>
print(be.title.text) #东小东页
print(be.title.string) #东小东页

#函数find_all()和find()使用,参数使用是相同的
#参数值均可使用：字符串、列表、正则对象、True（任意值）
import re #使用正则表达式
print(be.find_all(class_="yzm",limit=2))#limit为要返回的条数
print(be.find_all('input')) #查询所有标签名为input，存入到列表
be.find_all(id='link2')#通过id值查找
print(be.find_all(type=True))#type为true表示可以接收任意值
print(be.find_all(class_="yzm"))#通过class属性查找内容，注意class后面有下划线
print(be.find_all(src=re.compile(r"img/.*?jpg")))#通过src属性查找
print(be.find_all('img')[0]["src"])# img/0.jpg
#--------------------------------------------------
for inx in be.find_all(re.compile(r"i")):#匹配带i字母的所有标签名
    print(inx.name)
#------------------------------------------------
for inx in be.find_all(["input","img"]):#列表传递多个需匹配标签名
   print(inx)
   print(inx.get("name"))#获取标签name属性值
#------------------------------------------------------

#找到第一个，且只找一个
print(be.find(type="file"))#通过属性查找
print(be.find("input"))#通过标签查找
print(be.find("input",type="password"))#通过标签加属性查找，可支持有：id、type等
#参数不支持name和data-*
print(be.find_all(attrs={"name":"yzm"}))#可用此方法解决
