# -*- coding: utf-8 -*-
from argparse import ArgumentParser
from crawler.crawler import Crawler


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-t', '--threshold', type=float, default=0.6, help='crawling quality threshold (0-1)')
    parser.add_argument('-d', '--delay', type=float, default=0.05, help='crawling delay')
    parser.add_argument('-n', '--number', type=int, default=100000, help='crawling data number')
    args = parser.parse_args()

    crawler = Crawler(threshold=args.threshold, delay=args.delay)

    # Get number of crawling data for each section
    section_num = len(crawler.sections)
    data_num = {}
    for section in crawler.sections:
        data_num[section] = args.number // section_num
    data_num[crawler.sections[0]] += args.number - ((args.number // section_num) * section_num)

    # Crawl data for each section
    for section in crawler.sections:
        crawler.crawl(data_num=data_num[section], section=section)

    # Shuffle crawled dataset
    crawler.shuffle_dataset()
