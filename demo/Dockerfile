FROM python:3.8

COPY . /app

WORKDIR /app

COPY requirements.txt app/requirements.txt
RUN pip install -r app/requirements.txt

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN mkdir -p /root/.streamlit

RUN bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml'

RUN bash -c 'echo -e "\
[server]\n\
enableCORS = false\n\
" > /root/.streamlit/config.toml'

EXPOSE 8501

CMD streamlit run app.py --server.enableXsrfProtection=false