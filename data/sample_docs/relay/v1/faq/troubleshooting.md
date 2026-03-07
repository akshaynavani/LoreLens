---
title: Relay Troubleshooting FAQ
---
# Relay Troubleshooting FAQ

## Why is my consumer lag growing?

Consumer lag grows when consumers process slower than producers publish. Add consumers to
the group (up to one per partition), increase `max_poll_records`, or increase partitions.

## What does `REBALANCE_IN_PROGRESS` mean?

The consumer group is reassigning partitions, usually because a consumer joined, left, or
exceeded `session_timeout_ms` (default 45000). Keep processing per poll below the timeout.
