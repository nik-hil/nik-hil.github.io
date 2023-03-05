---
layout: post
title:  "Golang Journey Part 1"
date:   2023-03-05 2:24:42 +0530
tags:
 - go
 - tutorial
 - journey
---

## How did I start using Golang?

I initially started with Golang to work with Kubernetes code. I had the chance to work as a Kubernetes operator. To gain more expertise, I started looking at internet stories. How Golang is easy to adapt with very few key words and how easily it scales to multiple cores

I had gone through the tutorial given by Matt. It is a brilliant tutorial on Golang, and I would recommend this one. It had a similar style and passion to what I had seen in my MTech days (postgraduate).

I had to spend time writing sample code as mentioned in the tutorial. It was pretty difficult to understand the "sieve of Eratosthenes" algorithm in Golang (https://goplay.tools/snippet/NvVARuJ50Dz).

It creates a filter routine on the fly. The number passes through these filters, and if it reaches the end, it is a prime number. This prime number is added as a filter routine at the end.
Initially, it was difficult to imagine how this would work. The concept of a channel working independently with goroutine was a little surprising.


## Sample apps  

After going through various tutorials (https://github.com/nik-hil/Go-Tutorial), I jumped to a few sample applications. I developed a sample app using the Beego framework. I chose this because it had philosophysimiliar  to Django's.
Next, I got hold of the Gin frame work. It is good for starting your scalable web app without much fuss.
In another project, I had the chance to look at various options for connecting to databases. I had to choose between the standard library, sqlx, and gorm. I decided to go with Gorm based on its popularity.


## Golang Features

Goroutine and the channel are favourite features of many gophers. I really like the idea of CGO. It allows Go code to call C libraries. Testing is very easy. Table-driven tests help you gather multiple scenarios into one test case. Fuzzy testing, benchmarking, and profiling-related inbuilt tools are really helpful in scalable apps.


## Golang libraries

Gophers are divided on which library to use for a purpose. It looks like, for web development and database management, they prefer standard libraries. But for logging, there are so many favourite libraries. I think one major problem with the Go ecosystem is the lack of famous libraries. Many libraries are started with great promise, they are adapted by gophers, and suddenly they are abandoned for all the right reasons. This makes everyone hesitant to use unpopular libraries.



