# Nimbus Assistant HTTP API

## POST /v1/chat/completions

Creates a chat completion. The request and response shapes follow the OpenAI
chat completions API.

### Parameters

- `model` (required): model identifier, for example `nimbus`.
- `messages` (required): array of messages. Every message has a `role`
  (`system`, `user`, or `assistant`) and `content`.
- `max_tokens` (optional): maximum number of generated tokens. Default: 512.
- `temperature` (optional): sampling temperature between 0 and 1. Default: 0.7.
- `stream` (optional): when `true`, the response is streamed as server-sent
  events.

### Example

```bash
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"nimbus","messages":[{"role":"user","content":"Hello"}]}'
```

### Response

The JSON response contains a `choices` array. Each entry has a `message`
object whose `content` field holds the answer.

## GET /health

Returns service health and the loaded model name. Use this endpoint when
diagnosing connection failures.

## GET /v1/models

Lists the models exposed by the service.
