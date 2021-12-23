# Abstractive-Summarizer

KAIST 2020 Fall Semester - CS409: Software Projects for Industrial Collaboration

20160394 Sangbaek Yoo, 20160500 Jaeho Lee, 20160580 Sanghyun Jung

## 개발 환경
* Windows 10 - 64bit
* Ubuntu 16.04
* Python 3.7

## 실행 방법
### 데이터 크롤러
```
python crawl.py -t 0.6 -d 0.05 -n 100000
```
* ```-t``` quality threshold | default: 0.6
* ```-d``` crawling delay | default: 0.05
* ```-n``` crawling number | default: 100000

### 모델 학습
```
python train.py -t sentencepiece -s transformer -e 20 -b 128
```
* ```-t``` tokenizer (okt, sentencepiece) | default: sentencepiece
* ```-s``` seq2seq model (rnn, transformer) | default: transformer
* ```-e``` train epochs | default: 20
* ```-b``` batch size | default: 128

### 모델 평가
```
python evaluate.py -t sentencepiece -s transformer -b 256
```
* ```-t``` tokenizer (okt, sentencepiece) | default: sentencepiece
* ```-s``` seq2seq model (rnn, transformer) | default: transformer
* ```-b``` batch size | default: 256
* ```-e``` eval file | default: ./dataset/ncsoft/2020.09.18/eval.csv
* ```-a``` answer file | default: ./dataset/ncsoft/2020.09.18/answer.txt

### 모델 예측
```
python predict.py -t sentencepiece -s transformer
```
* ```-t``` tokenizer (okt, sentencepiece) | default: sentencepiece
* ```-s``` seq2seq model (rnn, transformer) | default: transformer

### 서버 구동
```
python server.py -t sentencepiece
```
* ```-t``` tokenizer (okt, sentencepiece) | default: sentencepiece

## 참고 논문
* ["Neural Machine Translation of Rare Words with Subword Units"](https://arxiv.org/abs/1508.07909)
* ["Effective Approaches to Attention-based Neural Machine Translation"](https://arxiv.org/abs/1508.04025)
* ["Attenttion Is All You Need"](https://arxiv.org/abs/1706.03762)
