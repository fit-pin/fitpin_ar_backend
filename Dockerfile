# ubuntu 베이스 이미지 사용
FROM da864268/my-ubuntu

LABEL maintainer="da864268@naver.com"
LABEL description="fitpin"

# bash 로 변경
SHELL ["/bin/bash", "-c"]

# 패키지 설치 
RUN apt update && apt upgrade && apt install \
libgl1 libglib2.0-0 -y

#Conda 설치
RUN mkdir -p ~/miniconda3 && \
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && \
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && \
rm -rf ~/miniconda3/miniconda.sh && \
source ~/.bashrc

ENV PATH "/root/miniconda3/bin:$PATH"
RUN source ~/.bashrc && conda init

#fitpin_ar_backend clone
RUN git clone https://github.com/fit-pin/fitpin_ar_backend.git
WORKDIR "/workspace/fitpin_ar_backend"

# conda 가상환경 만들기
RUN conda env create -p .conda && chmod +x start.sh

# 컨테이너 시작시 start.sh 파일 실행
CMD ["/bin/bash", "-c", "./start.sh"]