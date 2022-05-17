FROM python:3.7

COPY requirements.txt .

ENV LANG en_US.UTF-8 
ENV BOSDYN_CLIENT_USERNAME=Sasha
ENV BOSDYN_CLIENT_PASSWORD=olympicsasha1

COPY draw_circles.py app/
RUN pip install -r requirements.txt

WORKDIR app/

ENTRYPOINT ["/usr/bin/python3", "/app/draw_circles.py",  "192.168.80.3"]
