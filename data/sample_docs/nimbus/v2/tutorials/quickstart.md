---
title: Quickstart
---
# Quickstart

This tutorial uploads your first file to Nimbus in under five minutes.

## Install the SDK

```bash
pip install nimbus-sdk==2.*
```

## Upload a file

```python
from nimbus import Client

client = Client.from_env()  # reads NIMBUS_CLIENT_ID / NIMBUS_CLIENT_SECRET
bucket = client.buckets.create("quickstart-demo", region="us-east-1")
bucket.upload("hello.txt", b"hello nimbus")
print(bucket.list())
```

## Next steps

- Read the Authentication guide to choose scopes.
- Enable versioning on buckets that store user content.
