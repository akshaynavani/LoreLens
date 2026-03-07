---
title: Authentication (legacy)
tags: [auth, security, legacy]
---
# Authentication (legacy)

Nimbus v1 authenticates every request with a static API key sent in the `X-Nimbus-Key`
header. API keys never expire; rotate them manually from the console.

```bash
curl https://api.nimbus.dev/v1/buckets -H "X-Nimbus-Key: $NIMBUS_API_KEY"
```

## Migrating to v2

Static keys are deprecated and will stop working on **2027-03-31**. Migrate to OAuth 2.0
client credentials as described in the v2 Authentication guide.
