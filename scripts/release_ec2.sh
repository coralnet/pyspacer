# Notes from releasing using an aws ec2 instance
# SSH in: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstancesLinux.html 
# Install docker: https://docs.aws.amazon.com/AmazonECS/latest/developerguide/docker-basics.html 
# Install aws cli2: https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2-linux.html
# Copy over ~/.aws/credentials

ENV=${1:-staging}

NAME=public.ecr.aws/w6h8w6v4/coralnet.spacer:${ENV}

/usr/local/bin/aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws
docker build -t ${NAME} .
docker push ${NAME}


