FROM postgres:latest
RUN apt-get update && apt-get install -y dos2unix
RUN apt-get install -yq python-pip python-dev build-essential
RUN pip install awscli
COPY pg_dump-to-s3 /home
RUN chmod +x /home/pg_dump-to-s3.sh /home/s3-autodelete.sh
RUN crontab /home/backup.txt
RUN service cron start
RUN dos2unix /home/pg_dump-to-s3.sh
RUN dos2unix /home/s3-autodelete.sh
