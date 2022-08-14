AWS Lambda
========================

AWS Lambda's security policy does not allow you to write anywhere outside `/tmp` directory.
There are two things you need to do to make `doctr` work on lambda:
1. Disable usage of `multiprocessing` package by setting `DOCTR_MULTIPROCESSING_DISABLE` enivronment variable to `TRUE`. You need to do this, because this package uses `/dev/shm` directory for shared memory.
2. Change directory `doctr` uses for caching models. By default it's `~/.cache/doctr` which is outside of `/tmp` on AWS Lambda'. You can do this by setting `DOCTR_CACHE_DIR` enivronment variable.
