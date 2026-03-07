---
title: REST API Reference
---
# REST API Reference

Base URL: `https://api.nimbus.dev/v2`. All requests require a bearer token.

## Buckets

### List buckets

`GET /buckets` returns up to 100 buckets per page. Use the `cursor` query parameter from
`next_cursor` to paginate.

```http
GET /v2/buckets?limit=50 HTTP/1.1
Authorization: Bearer <access_token>
```

### Create a bucket

`POST /buckets` with a JSON body. Bucket names must be 3-63 lowercase characters and
globally unique.

```json
{ "name": "media-assets", "region": "eu-west-1", "versioning": true }
```

## Objects

### Upload an object

`PUT /buckets/{bucket}/objects/{key}` uploads up to 5 GiB in a single request. For larger
files use multipart upload (`POST /buckets/{bucket}/uploads`), with parts between 5 MiB and
5 GiB and at most 10,000 parts.

## Rate limits

Each credential may issue **600 requests per minute**. Exceeding the limit returns
`429 rate_limited` with a `Retry-After` header in seconds. SDKs retry with exponential
backoff automatically.
