# Demo knowledge base

These documents are original sample content written for llama-kb. They are
licensed under the same MIT license as the rest of the repository and contain
no third-party documentation.

- `docs/product-manual.md` — English product manual (facts: port 8080, admin
  console on 9090).
- `docs/api-reference.md` — English HTTP API reference.
- `docs/gateway-runbook.md` — Chinese operations runbook (facts: management
  port 9100, 30-day audit logs).
- `docs/aurora-operations.md` — English runbook for cross-lingual retrieval
  (facts: `catalog.db`, six-hour backups, administration port 9200).

To use them:

```bash
mkdir -p ~/.llama-kb
cp -r examples/docs ~/.llama-kb/docs
cp examples/catalog.json ~/.llama-kb/catalog.json
python -m llama_kb.cli fetch-model
python -m llama_kb.cli build
```

The catalog defines two groups, `product` and `ops`. Try pinning a group with
`--doc` and comparing the results:

```bash
python -m llama_kb.cli search "knowledge base retrieval is inaccurate" --doc product
python -m llama_kb.cli search "How long are audit logs retained?" --doc ops
```
