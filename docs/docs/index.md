# ml-research-kills-alpha documentation!

## Description

We measure how much of the ML edge in predicting future stocks survives when accounting for availability of algorithm architectures

## Commands

The Makefile contains the central entry points for common tasks related to this project.

### Syncing data to cloud storage

* `make sync_data_up` will use `az storage blob upload-batch -d` to recursively sync files in `data/` up to `da_cambiare/data/`.
* `make sync_data_down` will use `az storage blob upload-batch -d` to recursively sync files from `da_cambiare/data/` to `data/`.


