ENV=${1:-staging}

NAME=public.ecr.aws/w6h8w6v4/coralnet.spacer:${ENV}

aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws
docker build -t ${NAME} .
docker push ${NAME}
