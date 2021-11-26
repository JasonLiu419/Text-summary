import codecs
import glob
import json
import random
import math

import numpy as np
import torch
import h5py

from tokenizer import FullTokenizer
from utils import clean_text_by_sentences
import re
import jieba
import json


class Dataset(object):

    def __init__(self, file_pattern = None, vocab_file = None):

        self._file_pattern = file_pattern
        self._max_len = 60
        print("input max len : "+str(self._max_len))
        if vocab_file is not None:
            self._tokenizer = FullTokenizer(vocab_file, True)

    def iterate_once_doc_tfidf(self):
        def file_stream():
            for file_name in glob.glob(self._file_pattern):
                yield file_name
        for value in self._doc_stream_tfidf(file_stream()):
            yield value

    def _doc_stream_tfidf(self, file_stream):
        for file_name in file_stream:
            for doc in self._parse_file2doc_tfidf(file_name):
                yield doc
    


    def _parse_file2doc_tfidf(self, file_name):
        #print("Processing file: %s" % file_name)

        # #最初版-------------------------------------------------
        # with h5py.File(file_name,'r') as f:
        #     for j_str in f['dataset']:
        #         obj = json.loads(j_str)
        #         article, abstract = obj['article'], obj['abstract']
        #         #article, abstract = obj['article'], obj['abstracts']
        #         #print(article)
        #         #print(abstract)
        #         clean_article = clean_text_by_sentences(article)
        #         segmented_artile = [sentence.split() for sentence in clean_article]
        #         #print(tokenized_article[0])

        #         yield article, abstract, [segmented_artile]
        # #-----------------------------------------------------

        ##中文txt读入----------------------------------------------------------------
        #从文件中读取出所有的text和summary，存到summary【】和text【】里
        data = open(file_name)
        #context = data.readlines()
        summary = []
        text = []
        line1 =""
        summarytotal=""
        texttotal=""
        line = data.readline()
        while line:
            while (line and line != "summary\n" and line != "text\n"):
                #print(1)
                if(line1 == "summary\n"):
                    summarytotal = summarytotal + line
                if(line1 == "text\n"):
                    texttotal = texttotal + line
                line = data.readline()
                #print(line)
            line1 = line
            line = data.readline()
            if(summarytotal != ''):
                summary.append(summarytotal)
                summarytotal = ''
            if(texttotal != ''):
                text.append(texttotal)
                texttotal = ''


        #print(summary)
        #print(text)


        #消除字符间的空格
        for k in range(int(len(summary))):
            txt1 = summary[k].split()
            summary[k] = "".join(txt1)
        for k in range(int(len(text))):
            txt1 = text[k].split()
            text[k] = "".join(txt1)
        #print(context)

        #print(context[0])
        #print(context[1])
        #i=0#分辨summary和text
        article =[]
        abstract = []
        #进行分句
        for i in range(int(len(summary))):
            segmented_artile=[]
            article = self._cut_sent(text[i])
            abstract = self._cut_sent(summary[i])
            #print(article)
            #print(abstract)
        # with h5py.File(file_name,'r') as f:
        #     for j_str in f['dataset']:
        #         obj = json.loads(j_str)
        #print(article)
        #segmented_artile = self._cut_sent(article)
        #article, abstract = obj['article'], obj['abstracts']
        #sg = ''.join(article)
        #segmented_artile = jieba.cut(sg)
        
            # #######################中文分词--------------------------------------  
            #stopword_list = self.get_stopword_list('/home/yuanben/py/data/stopwords-master/cn_stopwords.txt') 
            # txta = context[2*i].split()
            # txtb = "".join(txta)
            # #print(txtb)
            # segmented_artile  =[jieba.lcut(txtb)]
            for k in range(len(article)):
                #article[k] = self.ExchangeChar(article[k])
                #article[k]= self.clean_stopword(article[k],stopword_list)
                segmented_artile.append(jieba.lcut(article[k]))
                #segment_article中保存的是根据分词表进行分词过后的文本，是把article分词后的结果
            # for k in range(len(article)):
            #     infor = article[k]
            #     segmented_artile.append(jieba.lcut(infor))
            #print(segmented_artile)
            #print(article)
            #print(abstract)
            #----------------------------------------------------------------------

            #clean_article = clean_text_by_sentences(article)
            #segmented_artile = [sentence.split() for sentence in clean_article]
        #print(segmented_artile)
        #data.close()
        #print(tokenized_article[0])
            print("article")
            print(article)
            print("segmented_artile")
            print(segmented_artile)
            yield article, abstract, [segmented_artile]
        #----------------------------------------------------------------------------------------

    def _cut_sent(self,para):
        para = str(para)
        para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
        para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
        para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
        para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
        # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
        para = para.rstrip()  # 段尾如果有多余的\n就去掉它
        # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
        return para.split("\n")
    
    def ExchangeChar(self, context,IsChinese=False):
        ChineseChar=['，','！','。','：','《','》','（','）','？','“','”','、','|','‘','’','的','']   #中文标点符号大概是15种
        EnglishChar=['','','','','','','','','','','','','','','']   #要互换的英文标点符号，与上面的中文列表一 一对应哦
        for k in range(len(EnglishChar)):
            if IsChinese==True:
                context = context.replace(EnglishChar[k],ChineseChar[k])
            else:
                context = context.replace(ChineseChar[k],EnglishChar[k])
        return context

    # 读取停用词列表
    def get_stopword_list(self,file):
        with open(file, 'r', encoding='utf-8') as f:    # 
            stopword_list = [word.strip('\n') for word in f.readlines()]
        return stopword_list


    # 分词 然后清除停用词语
    def clean_stopword(self,str, stopword_list):
        result = ''
        word_list = jieba.lcut(str)   # 分词后返回一个列表  jieba.cut(）   返回的是一个迭代器
        for w in word_list:
            if w not in stopword_list:
                result += w
        return result

    # def _parse_file2doc_tfidf(self, file_name):
    #     print("Processing file: %s" % file_name)
    #     with h5py.File(file_name,'r') as f:
    #         for j_str in f['dataset']:
    #             obj = json.loads(j_str)
    #             article, abstract = obj['article'], obj['abstract']
    #             #article, abstract = obj['article'], obj['abstracts']
    #             clean_article = clean_text_by_sentences(article)
    #             segmented_artile = [sentence.split() for sentence in clean_article]
    #             #print(tokenized_article[0])

    #             yield article, abstract, [segmented_artile]


    def iterate_once_doc_bert(self):
        def file_stream():
            for file_name in glob.glob(self._file_pattern):
                yield file_name
        for value in self._doc_iterate_bert(self._doc_stream_bert(file_stream())):
            yield value

    def _doc_stream_bert(self, file_stream):
        for file_name in file_stream:
            for doc in self._parse_file2doc_bert(file_name):
                yield doc

    def _parse_file2doc_bert(self, file_name):
        print("Processing file: %s" % file_name)
        # with h5py.File(file_name,'r') as f:
        #     for j_str in f['dataset']:
        #         obj = json.loads(j_str)
        #         article, abstract = obj['article'], obj['abstract']
        #中文txt读取——————————————————————————————————————————
        data = open(file_name)
        #context = data.readlines()
        summary = []
        text = []
        line1 =""
        summarytotal=""
        texttotal=""
        line = data.readline()
        while line:
            while (line and line != "summary\n" and line != "text\n"):
                #print(1)
                if(line1 == "summary\n"):
                    summarytotal = summarytotal + line
                if(line1 == "text\n"):
                    texttotal = texttotal + line
                line = data.readline()
                #print(line)
            line1 = line
            line = data.readline()
            if(summarytotal != ''):
                summary.append(summarytotal)
                summarytotal = ''
            if(texttotal != ''):
                text.append(texttotal)
                texttotal = ''


        #print(summary)
        #print(text)


        #消除字符间的空格
        for k in range(int(len(summary))):
            txt1 = summary[k].split()
            summary[k] = "".join(txt1)
        for k in range(int(len(text))):
            txt1 = text[k].split()
            text[k] = "".join(txt1)
        #print(context)

        #print(context[0])
        #print(context[1])
        #i=0#分辨summary和text
        article =[]
        abstract = []
        #进行分句
        for i in range(int(len(summary))):
            segmented_artile=[]
            article = self._cut_sent(text[i])
            abstract = self._cut_sent(summary[i])
                #article, abstract = obj['article'], obj['abstracts']
            tokenized_article = [self._tokenizer.tokenize(sen) for sen in article]
            #print(tokenized_article[0])

            article_token_ids = []
            article_seg_ids = []
            article_token_ids_c = []
            article_seg_ids_c = []
            pair_indice = []
            k = 0
            for i in range(len(article)):
                for j in range(i+1, len(article)):

                    tokens_a = tokenized_article[i]
                    tokens_b = tokenized_article[j]

                    input_ids, segment_ids = self._2bert_rep(tokens_a)
                    input_ids_c, segment_ids_c = self._2bert_rep(tokens_b)
                    assert len(input_ids) == len(segment_ids)
                    assert len(input_ids_c) == len(segment_ids_c)
                    article_token_ids.append(input_ids)
                    article_seg_ids.append(segment_ids)
                    article_token_ids_c.append(input_ids_c)
                    article_seg_ids_c.append(segment_ids_c)

                    pair_indice.append(((i,j), k))
                    k+=1
            yield article_token_ids, article_seg_ids, article_token_ids_c, article_seg_ids_c, pair_indice, article, abstract

    def _doc_iterate_bert(self, docs):

        for article_token_ids, article_seg_ids, article_token_ids_c, article_seg_ids_c, pair_indice, article, abstract in docs:

            if len(article_token_ids) == 0:
                yield None, None, None, None, None, None, pair_indice, article, abstract
                continue
            num_steps = max(len(item) for item in article_token_ids)
            #num_steps = max(len(item) for item in iarticle)
            batch_size = len(article_token_ids)
            x = np.zeros([batch_size, num_steps], np.int32)
            t = np.zeros([batch_size, num_steps], np.int32)
            w = np.zeros([batch_size, num_steps], np.uint8)

            num_steps_c = max(len(item) for item in article_token_ids_c)
            #num_steps = max(len(item) for item in iarticle)
            x_c = np.zeros([batch_size, num_steps_c], np.int32)
            t_c = np.zeros([batch_size, num_steps_c], np.int32)
            w_c = np.zeros([batch_size, num_steps_c], np.uint8)
            for i in range(batch_size):
                num_tokens = len(article_token_ids[i])
                x[i,:num_tokens] = article_token_ids[i]
                t[i,:num_tokens] = article_seg_ids[i]
                w[i,:num_tokens] = 1

                num_tokens_c = len(article_token_ids_c[i])
                x_c[i,:num_tokens_c] = article_token_ids_c[i]
                t_c[i,:num_tokens_c] = article_seg_ids_c[i]
                w_c[i,:num_tokens_c] = 1

            if not np.any(w):
                return
            out_x = torch.LongTensor(x)
            out_t = torch.LongTensor(t)
            out_w = torch.LongTensor(w)

            out_x_c = torch.LongTensor(x_c)
            out_t_c = torch.LongTensor(t_c)
            out_w_c = torch.LongTensor(w_c)

            yield  article, abstract, (out_x, out_t, out_w, out_x_c, out_t_c, out_w_c, pair_indice)

    def _2bert_rep(self, tokens_a, tokens_b=None):

        if tokens_b is None:
            tokens_a = tokens_a[: self._max_len - 2]
        else:
            self._truncate_seq_pair(tokens_a, tokens_b, self._max_len - 3)

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b is not None:

            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)

            tokens.append("[SEP]")
            segment_ids.append(1)
        #print(tokens)
        input_ids = self._tokenizer.convert_tokens_to_ids(tokens)
        #print(input_ids)

        return input_ids, segment_ids

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()
