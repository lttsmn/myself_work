#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-05-05 22:15:13
# @Author  : JackPI (1129501586@qq.com)
# @Link    : https://blog.csdn.net/meiqi0538
# @Version : $Id$
#导入jieba包
#https://blog.csdn.net/meiqi0538/article/details/80218870
import jieba
#管理系统路径
import sys
#获取自定义词典
jieba.load_userdict("./user_dict/userdict_food.txt")
#导入词性标注的包
import jieba.posseg as pseg

#添加词
jieba.add_word('石墨烯')
jieba.add_word('凱特琳')
#删除词
jieba.del_word('自定义词')
#元组类型的测试数据
test_sent = (
"李小福是创新办主任也是云计算方面的专家; 什么是八一双鹿\n"
"例如我输入一个带“韩玉赏鉴”的标题，在自定义词库中也增加了此词为N类\n"
"「台中」正確應該不會被切開。mac上可分出「石墨烯」；此時又可以分出來凱特琳了。"
)
#默认分词
words = jieba.cut(test_sent)
print('/'.join(words))#使用/把分词的结果分开

print("="*40)
#用于词性标注
result = pseg.cut(test_sent)
#使用for循环把分出的词及其词性用/隔开，并添加，和空格
for w in result:
    print(w.word, "/", w.flag, ", ", end=' ')

print("\n" + "="*40)

#对英文的分割
terms = jieba.cut('easy_install is great')
print('/'.join(terms))
#对英文和汉字的分割
terms = jieba.cut('python 的正则表达式是好用的')
print('/'.join(terms))

print("="*40)
# test frequency tune
testlist = [
('今天天气不错', ('今天', '天气')),
('如果放到post中将出错。', ('中', '将')),
('我们中出了一个叛徒', ('中', '出')),
]

for sent, seg in testlist:
    print('/'.join(jieba.cut(sent, HMM=False)))
    word = ''.join(seg)
    print('%s Before: %s, After: %s' % (word, jieba.get_FREQ(word), jieba.suggest_freq(seg, True)))
    print('/'.join(jieba.cut(sent, HMM=False)))
    print("-"*40)