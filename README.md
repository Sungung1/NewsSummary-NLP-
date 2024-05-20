# NewsSummary-NLP-
News Summary using kobart, bert, our model

이 프로젝트는 조별 과제로 개발된 뉴스 기사 요약 및 레포트 생성 웹 애플리케이션입니다. AI-Hub에서 학습 데이터를 가져와 BERT, KoBART와 맞춤형 Seq2Seq 요약 모델을 활용하여 뉴스 기사를 요약합니다. 이 애플리케이션은 플라스크(Flask) 프레임워크를 사용하여 사용자에게 간편한 URL 입력과 요약문 생성을 제공합니다.

데이터 구조 및 학습
데이터 출처: AI-Hub
학습 데이터: 17,300 파일
검증 데이터: 4,300 파일

요약 모델
Seq2Seq 모델:
모델 구조: Attention 메커니즘을 포함한 인코더-디코더 구조
전처리: 불용어 제거, 데이터 분리, 토큰화, 정수 인코딩
학습 및 평가: 학습 과정에서 생성된 요약 모델을 통해 텍스트를 요약하고 ROUGE metric을 사용하여 성능 평가

ROUGE 성능:
ROUGE-1: 0.041
ROUGE-2: 0.017
ROUGE-L: 0.041

사전 훈련된 모델:
BERT: 사전 훈련된 BERT 모델을 사용하여 텍스트 요약
KoBART: 사전 훈련된 KoBART 모델을 사용하여 한국어 텍스트 요약
