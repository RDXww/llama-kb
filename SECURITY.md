# Security

## Threat model

llama-kb is designed for **single-user, local deployment**. The RAG proxy has
no authentication and no authorization layer. It binds to `127.0.0.1` by
default. Exposing it to a LAN or the internet without an authenticating reverse
proxy is out of scope and dangerous: anyone who can reach the port can query
your indexed documents.

The launcher starts `llama-server` on loopback as well. That server is also
unauthenticated.

## Built-in protections

- Loopback binding by default, with an explicit warning when changed.
- No CORS enablement and no arbitrary URL forwarding; the upstream URL is fixed
  by configuration.
- Request body size limits and JSON validation.
- Retrieved documents are treated as untrusted data in the system prompt, and
  the model is instructed to ignore embedded instructions.
- Source payloads are size-limited so response headers stay valid.
- The build lock prevents concurrent writers to the same index.

Prompt-injection mitigations reduce risk but do not eliminate it. Assume a
determined document can still influence model output.

## Deploying on a network

If you must expose the proxy:

1. Put it behind a reverse proxy with authentication, rate limiting, and TLS.
2. Keep the upstream model server reachable only from the proxy.
3. Set a strict request body limit and monitor usage.
4. Treat the knowledge base as sensitive: the proxy is a query interface to it.

## Reporting a vulnerability

Please do not open a public issue for security problems. Use GitHub's private
security advisory feature on the repository instead. Include reproduction
steps and the impact you believe is possible.
