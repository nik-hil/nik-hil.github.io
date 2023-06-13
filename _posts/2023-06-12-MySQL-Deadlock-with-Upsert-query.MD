---
layout: post
title:  MySQL deadlock with Upsert query
date:   2023-06-12 19:24:42 +0530
tags: MySQL Deadlock Golang
---

* toc
{:toc}


## Introduction:
In high-performance database systems, deadlocks can sometimes occur, leading to errors and degraded performance. This blog post aims to shed light on the scenario where upsert operations in MySQL result in "Error 1213: Deadlock found when trying to get lock." We will explore the reasons behind this error, the impact of varying workload sizes, and provide strategies for resolving deadlocks and optimizing performance.

## How did I came across this problem?
When performing upsert (insert or update) queries with MySQL, I encountered a scenario where the following observations were made:

Upserting 100 rows in the database multiple times within a short span resulted in the occurrence of "Error 1213: Deadlock found when trying to get lock." Additionally, the MySQL CPU utilization was observed to be around 30%.

Conversely, upserting 10 rows in the database multiple times within a short span did not result in any deadlock errors. However, the MySQL CPU utilization was significantly higher, approximately 97%.

## Analysis

A deadlock is a situation in which two or more transactions are waiting for each other to release a lock. This can happen when two transactions are trying to access the same data at the same time. In MySQL, each row is locked when it is being updated or inserted. This prevents other transactions from accessing the row until the current transaction is complete.

When I upsert 100 rows in MySQL, you are essentially performing 100 separate transactions. This means that there is a high chance that two or more of these transactions will try to access the same row at the same time. This can lead to a deadlock, which will cause the error message "Error 1213: Deadlock found when trying to get lock" to be displayed.

## CPU Utilization

Inserting 100 rows in gave better CPU utilization. 

## Resolving Deadlocks and Improving Performance

Without MySQL server, I have opted for using retry mechanism to do upsert. Sample (untested) code in Golang with Gorm,
```Golang
func Publish(rows []Message) {
	start := time.Now()
	maxRetries := 3
	retryInterval := 2
	for retries := 0; retries < maxRetries; retries++ {
		result := BulkUpsert(rows)
		if result.Error != nil {
			if shouldRetry(result.Error) {
				time.Sleep(time.Second * time.Duration(retryInterval*retries+1))
				continue
			}
		}
		retries = maxRetries
		if result.RowsAffected > 0 {
		}
	}
}
func BulkUpsert(rows []Message) *gorm.DB {
	// execute upsert query
}
func shouldRetry(err error) bool {
	if mysqlErr, ok := err.(*mysql.MySQLError); ok {
		if mysqlErr.Number == 1213 { // deadlock
			return true
		}
	}
	return false
}
```
Other ways can be to go for Postgres or Oracle DB. 

[Note: This blog post is intended to provide insights and general guidance. It is recommended to consult the official MySQL documentation and seek expert advice for specific deadlock scenarios and complex database setups.]