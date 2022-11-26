---
layout: post
title:  "Golang error handling"
date:   2022-11-27 2:24:42 +0530
tags:
 - go
 - golang
 - error
---


## Different ways in which we can handle errors in Golang.

### Using Go default error handling
Golang error handling is different from the traditional programming language.

```go
// https://goplay.tools/snippet/1FBSURLzkxb
package main

import (
	"errors"
	"fmt"
	"log"
)

func A() error {
	return errors.New("A error")
}

func B() error {
	return fmt.Errorf("Call A(), %+v", A())
}
func main() {
	fmt.Println("Hello World")
	err := B()
	if err != nil {
		log.Printf("Call B(), %+v", err)
	}
}
```

In `func A()` we are creating a new Error with `errors.New`. This is a custom error. 
In `func B()` we are again receiving an error. Now either we can log this or send error back to the caller. We did create a new error using 
`fmt.Errorf`.
In `func main()` we are logging the error.
Output will look like this,
```bash
Hello World
2022/11/26 17:55:05 Call B(), Call A(), A error
```

### Using pkg/errors

```go
// https://goplay.tools/snippet/CThylnUxj4i
package main

import (
	"fmt"
	"log"
	"github.com/pkg/errors"
)

func A() error {
	return errors.New("A error")
}

func B() error {
	return errors.Wrap(A(), "Call A()")
}
func main() {
	fmt.Println("Hello World")
	err := B()
	if err != nil {
		log.Printf("Call B(), %+v", err)
	}
}
```

We are creating a new error with this package `github.com/pkg/errors`. `errors.Wrap()` will add the previous error with the stack trace.
While printing the error, `%+v` will add the stack trace.

```bash
Hello World
[T+0000ms]
2009/11/10 23:00:00 Call B(), A error
main.A
	/tmp/sandbox2422463901/prog.go:12
main.B
	/tmp/sandbox2422463901/prog.go:16
main.main
	/tmp/sandbox2422463901/prog.go:21
runtime.main
	/usr/local/go-faketime/src/runtime/proc.go:250
runtime.goexit
	/usr/local/go-faketime/src/runtime/asm_amd64.s:1594
Call A()
main.B
	/tmp/sandbox2422463901/prog.go:16
main.main
	/tmp/sandbox2422463901/prog.go:21
runtime.main
	/usr/local/go-faketime/src/runtime/proc.go:250
runtime.goexit
	/usr/local/go-faketime/src/runtime/asm_amd64.s:1594
[T+0001ms]

```
