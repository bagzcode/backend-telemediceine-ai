#!/bin/bash

usage() {
  echo "Usage: $0 -s <server address> -p <port>" 1>&2;
  echo ""
  echo "E.g. $0 -s 172.24.0.1 -p 80"
  echo ""
  echo "       -s ..... server address"
  echo "       -p ..... forwarded port"
  exit 1;
}


while getopts "s:p:" opt
do
   case "$opt" in
      s ) SERVER_ADDRESS_TO_BE_REPLACED=$OPTARG ;;
      p ) PORT_TO_BE_REPLACED=$OPTARG ;;
      ? ) usage ;;
   esac
done

if [ -z "$SERVER_ADDRESS_TO_BE_REPLACED" ] || [ -z "$PORT_TO_BE_REPLACED" ] 
then
   echo "Some or all of the parameters are empty";
   usage
fi

cp docker-compose.yml.default docker-compose.yml

# fresh installation
echo ">> Fresh installation"
sudo docker system prune --all

echo ">> Injecting server's IP address"
sed -i "s/SERVER_ADDRESS_TO_BE_REPLACED/$SERVER_ADDRESS_TO_BE_REPLACED/g" ./docker-compose.yml
echo ">> Injecting forwarded port"
sed -i "s/PORT_TO_BE_REPLACED/$PORT_TO_BE_REPLACED/g" ./docker-compose.yml

echo ">> Running docker-compose"
sudo docker-compose up --build -d
