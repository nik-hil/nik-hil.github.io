---
layout: post
title:  "Simple Go app with docker"
date:   2022-06-05 2:24:42 +0530
tags:
 - go
 - docker
---


This will show how to use docker to create the container with go without using any other base image. Is creating an app from scratch is better than using any base image?

In which scenarios we should use the scratch and base image for our containers?

[Github repo](https://github.com/nik-hil/go-docker)

There are two Docker file in git repo.

[Dockerfile](https://github.com/nik-hil/go-docker/blob/master/Dockerfile) will build from the scratch. 
```
FROM scratch

EXPOSE 8080
COPY ./hello /
COPY templates templates

CMD ["/hello"]
```
[Dockerfile.multistage](https://github.com/nik-hil/go-docker/blob/master/Dockerfile.multistage) will use a base image and multiple docker layers. 
```
# Multi-stage build step 0
FROM golang:1.8

RUN mkdir -p /go/src/app
WORKDIR /go/src/app
COPY . /go/src/app

ENV CGO_ENABLED=0

RUN go-wrapper download
RUN go-wrapper install

# Multi-stage build step 1
FROM scratch
COPY --from=0 /go/bin/app /
COPY --from=0 /go/src/app/templates /templates

CMD ["/app"] 
```
Using multiple docker layer also creates the image of small size.

```
 $ docker image ls | grep -i godocker     
godocker                      scratch                                    6cd7ef9585e7   11 minutes ago   8.52MB
godocker                      multistage                                 5a70d6439286   19 minutes ago   7.79MB
```

## From scratch

`Dockerfile` will build from the scratch. Once we run this image, we wont be able to get a shell inside it. Because there is no shell.
We have to use another way (`docker container export`) to get the internal of container.

```
 ~  $ docker container ls
CONTAINER ID   IMAGE              COMMAND    CREATED          STATUS          PORTS                    NAMES
b945d0c0652d   godocker:scratch   "/hello"   38 seconds ago   Up 37 seconds   0.0.0.0:8080->8080/tcp   frosty_chandrasekhar

 ~  $ docker container export -o output frosty_chandrasekhar
 ~  $  tar -xvf output
x .dockerenv
x dev/
x dev/console
x dev/pts/
x dev/shm/
x etc/
x etc/hostname
x etc/hosts
x etc/mtab
x etc/resolv.conf
x hello
x proc/
x sys/
x templates/
x templates/pages.html
 ~  $ 
```


Building a docker image from scratch has atleast 2 pros.

1. Image size is small. 
1. Image has very small attack area. By excluding shell from docker image we have secured our container.


## Resources

1. https://github.com/lizrice/hello-container-world/tree/master/Example1
