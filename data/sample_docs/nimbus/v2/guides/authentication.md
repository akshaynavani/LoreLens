---
title: Authentication
tags: [auth, security]
---
# Authentication

Nimbus v2 uses OAuth 2.0 client credentials for server-to-server calls and short-lived
access tokens for everything else.

## Creating API credentials

1. Open **Console > Settings > API Credentials**.
2. Click **New credential** and pick the scopes your service needs.
3. Copy the `client_id` and `client_secret`. The secret is shown only once.

## Requesting an access token

Exchange your credentials at the token endpoint:

```bash
curl -X POST https://auth.nimbus.dev/v2/oauth/token \
  -d grant_type=client_credentials \
  -d client_id=$NIMBUS_CLIENT_ID \
  -d client_secret=$NIMBUS_CLIENT_SECRET \
  -d scope="storage:read storage:write"
```

The response contains an `access_token` that expires after **3600 seconds**.

## Refreshing tokens

Client-credential tokens cannot be refreshed. Request a new token when you receive a
`401 token_expired` error, or proactively 60 seconds before `expires_in` elapses. The
official SDKs do this automatically.

## Scopes

| Scope | Grants |
|-------|--------|
| `storage:read` | List and download objects |
| `storage:write` | Upload and delete objects |
| `admin` | Manage buckets, keys and billing |

Requests made with a token that lacks the required scope fail with `403 insufficient_scope`.
