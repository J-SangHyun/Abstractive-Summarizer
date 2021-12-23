# -*- coding: utf-8 -*-
import os
import kss
import csv
import json
import time
import datetime
import pandas as pd
from tqdm import tqdm
from crawler.utils import *


class Crawler(object):
    def __init__(self, threshold, delay):
        self.base_url = 'https://news.naver.com/main/list.nhn?mode=LSD&mid=shm&listType=summary'
        self.sections = ['정치', '경제', '사회', '생활/문화', 'IT/과학']
        self.sid1 = {'정치': 100, '경제': 101, '사회': 102, '생활/문화': 103, 'IT/과학': 105}

        today = date.today()
        self.threshold = threshold
        self.delay = delay
        self.log_file_path = './crawler/crawl_log.txt'
        self.dataset_dir = f'./dataset/crawled/{str(today).replace("-", ".")}/'
        self.dataset_path = os.path.join(self.dataset_dir, 'train.csv')

        self.log_data = {}
        for field in self.sections:
            self.log_data[field] = date_to_int(today)

        if os.path.exists(self.log_file_path):
            self.load_log()
        else:
            self.save_log()

        if not os.path.exists('./dataset/crawled/'):
            os.mkdir('./dataset/crawled/')

        if not os.path.exists(self.dataset_dir):
            os.mkdir(self.dataset_dir)

        if not os.path.exists(self.dataset_path):
            self.csv_initialize()

    def crawl(self, data_num, section):
        crawled_data_num = 0
        pbar = tqdm(total=data_num)
        pbar.set_description(f'{section} 섹션 뉴스기사 크롤링 중...')
        while True:
            # Date iteration
            crawl_date = int_to_date(self.log_data[section]) - datetime.timedelta(days=1)
            self.log_data[section] = date_to_int(crawl_date)
            self.save_log()
            date_titles = []
            date_articles = []

            # Get max page
            url = self.base_url + f'&sid1={self.sid1[section]}&date={date_to_int(crawl_date)}&page={10000}'
            max_page_bs = get_encoded_bs(url=url)
            max_page = int(max_page_bs.find(id='main_content').find('div', 'paging').find('strong').get_text())

            for page in range(1, max_page + 1):
                # Page iteration
                url = self.base_url + f'&sid1={self.sid1[section]}&date={date_to_int(crawl_date)}&page={page}'
                page_bs = get_encoded_bs(url=url)
                headline_list = []
                try:
                    headline_list += page_bs.find('ul', 'type06_headline').find_all('li')
                    headline_list += page_bs.find('ul', 'type06').find_all('li')
                except AttributeError:
                    pass

                for headline in headline_list:
                    # Article iteration
                    news_url = headline.find_all('dt')[-1].find('a').attrs['href']
                    article_id = int(news_url.split('&')[-1].split('=')[1])
                    news_bs = get_encoded_bs(url=news_url)
                    try:
                        title = news_bs.find(id='articleTitle').get_text()
                        title = remove_text_inside_brackets(title).strip()
                        if title in date_titles:
                            continue
                    except AttributeError:
                        continue

                    for tag in news_bs('a') + news_bs('span') + news_bs('script') + news_bs('strong'):
                        tag.decompose()
                    body_list = news_bs.find(id='articleBodyContents').find_all(text=True)

                    main_sentence, max_coverage = '', -1.0
                    for body in body_list:
                        stop_words = ['본문 내용', 'TV플레이어', '// TV플레이어', '동영상 뉴스', '//', '=', '◇', '◆',
                                      '●', '■', '□', '☆', '★', '※', '△', '▲', '▷', '▶', '▽', '▼', '◁', '◀',
                                      '// flash 오류를 우회하기 위한 함수 추가', 'function _flash_removeCallback() {}']
                        for stop_word in stop_words:
                            body = body.replace(stop_word, '')
                        body = remove_text_inside_brackets(body).strip()

                        body = kss.split_sentences(body)
                        for sentence in body:
                            splited_sentence = sentence.split()
                            if len(splited_sentence) < 6:
                                continue
                            if splited_sentence[1] == '기자' or splited_sentence[1] == '특파원':
                                sentence = ' '.join(splited_sentence[2:])
                            if splited_sentence[2] == '기자' or splited_sentence[2] == '특파원':
                                sentence = ' '.join(splited_sentence[3:])
                            try:
                                coverage = compute_noun_coverage(title, sentence)
                                if coverage > max_coverage:
                                    main_sentence = sentence
                                    max_coverage = coverage
                            except ValueError:
                                continue

                    if max_coverage > self.threshold and len(title) < len(main_sentence):
                        if main_sentence in date_articles:
                            continue
                        date_titles.append(title)
                        date_articles.append(main_sentence)
                        self.csv_add_data(article_id, title, main_sentence)
                        crawled_data_num += 1
                        pbar.update(1)
                        if crawled_data_num >= data_num:
                            pbar.close()
                            return
                    time.sleep(self.delay)

    def save_log(self):
        with open(self.log_file_path, 'w', encoding='utf-8') as f:
            json.dump(self.log_data, f, indent=4, ensure_ascii=False)

    def load_log(self):
        with open(self.log_file_path, 'r', encoding='utf-8') as f:
            self.log_data = json.load(f)

    def csv_initialize(self):
        with open(self.dataset_path, 'w', encoding='utf-8') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(['', 'inputs', 'targets'])

    def csv_add_data(self, article_id, title, main_sentence):
        with open(self.dataset_path, 'a', encoding='utf-8') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow([article_id, main_sentence, title])

    def shuffle_dataset(self):
        corpus = pd.read_csv(self.dataset_path, encoding='utf-8', index_col=[0])
        corpus = corpus.sample(frac=1)
        corpus.to_csv(self.dataset_path)
