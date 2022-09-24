FROM python:3.8.13

RUN apt-get update
RUN apt install -y locales && localedef -f UTF-8 -i ja_JP ja_JP.UTF-8
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8
ENV TZ JST-9

RUN apt-get -y upgrade && \
    apt-get install -y fonts-noto-cjk

RUN pip install --upgrade pip
RUN pip install notebook matplotlib seaborn pandas scikit-learn
RUN pip install "snowflake-snowpark-python[pandas]"

RUN mkdir -p /root/.config/matplotlib
ADD matplotlibrc /root/.config/matplotlib/matplotlibrc

WORKDIR /workspace

ENTRYPOINT ["jupyter"]
CMD ["notebook", "--port=8000", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''"]
