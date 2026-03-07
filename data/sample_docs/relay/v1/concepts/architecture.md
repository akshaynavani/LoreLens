---
title: Relay Architecture Overview
---
# Relay Architecture Overview

Relay is Nimbus' event streaming service. Producers publish events to **topics**; each
topic is split into **partitions** that are replicated three times across availability
zones.

## Delivery guarantees

Relay provides at-least-once delivery by default. Enable idempotent producers
(`enable_idempotence=true`) together with transactional consumers for exactly-once
processing within a single topic.

## Retention

Events are retained for 7 days by default. Retention can be raised to 90 days per topic,
or set to `compact` to keep only the latest event per key.
