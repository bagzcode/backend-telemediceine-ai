FROM nginx:1.19

COPY nginx.conf /etc/nginx/nginx.conf

USER root

ADD homepage /usr/share/nginx/html/

COPY ./entryPoint.sh /
RUN chmod +x entryPoint.sh

ENTRYPOINT ["/entryPoint.sh"]

CMD ["nginx", "-g", "daemon off;"]