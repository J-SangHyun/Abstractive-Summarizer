# -*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup
from bs4.dammit import EncodingDetector
from datetime import date
from konlpy.tag import Okt

okt = Okt()


def get_encoded_bs(url):
    res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    http_encoding = res.encoding if 'charset' in res.headers.get('content-type', '').lower() else None
    html_encoding = EncodingDetector.find_declared_encoding(res.content, is_html=True)
    encoding = html_encoding or http_encoding
    soup = BeautifulSoup(res.content, 'lxml', from_encoding=encoding)
    return soup


def compute_noun_coverage(title, sentence):
    """
    Calculate ratio of how many nouns in title covered by sentence.
    :param title: title
    :param sentence: sentence
    :return: noun coverage
    """
    title_nouns = okt.nouns(title)
    if len(title_nouns) == 0:
        return 0.0
    coverage = sum([1 if noun in sentence else 0 for noun in title_nouns]) / len(title_nouns)
    return coverage


def date_to_int(date):
    return 10000 * date.year + 100 * date.month + date.day


def int_to_date(int_date):
    return date(year=int_date//10000,
                month=(int_date % 10000)//100,
                day=int_date % 100)


def remove_text_inside_brackets(text, brackets="()[]<>【】"):
    count = [0] * (len(brackets) // 2)
    saved_chars = []
    for character in text:
        for i, b in enumerate(brackets):
            if character == b:
                kind, is_close = divmod(i, 2)
                count[kind] += (-1)**is_close
                if count[kind] < 0:
                    count[kind] = 0
                else:
                    break
        else:
            if not any(count):
                saved_chars.append(character)
    return ''.join(saved_chars)
