# AuroraDB Operations Guide

> Sample material for llama-kb. AuroraDB is a fictional database product.

## Metadata storage

AuroraDB stores all metadata in a single SQLite file named `catalog.db`.
Back up this file together with the data directory; restoring only the data
directory leaves the catalog inconsistent.

## Backup schedule

The backup interval is every six hours, and snapshots are retained for 30 days.
A manual snapshot can be triggered with `aurora snapshot create`.

## Administration endpoint

The administration endpoint listens on port 9200 by default and should only be
reachable from the internal network. Enable TLS before exposing it to other
subnets.
