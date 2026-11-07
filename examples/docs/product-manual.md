# Nimbus Assistant Product Manual

## Overview

Nimbus Assistant is a fictional, locally hosted AI workspace used as sample
material for llama-kb. The current release is version 2.5.1. It supports
multi-turn conversations, tool calls, and retrieval from a local knowledge
base.

## Deployment

### Docker

Run the container and publish the service on port 8080:

```bash
docker run -p 8080:8080 nimbus/assistant:2.5.1
```

### From source

```bash
make build
./assistant serve
```

The service listens on port 8080 by default. Set `NIMBUS_PORT` to change it.

## Troubleshooting

### The model cannot be connected

Check the following, in order:

1. Make sure port 8080 is free. Nimbus Assistant refuses to start when another
   process is already listening on 8080.
2. Confirm that the upstream model server is healthy by calling
   `GET /health` on the model host.
3. Check the launcher log; connection refused errors usually mean the model
   process exited during startup.

### Knowledge base retrieval is inaccurate

- Increase `top_k` to return more candidate chunks.
- Try a stronger multilingual embedding model.
- Make sure documents are indexed with the same embedding model used at query
  time; mixing models produces meaningless similarity scores.

## Administration

The administration console listens on port 9090 and binds to localhost by
default. It should never be exposed to the public internet without an
authentication proxy.
