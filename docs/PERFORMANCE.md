# Response time

Catalog data and images are local. OpenAI conversations still require network calls, so response times vary with the request and service conditions.

The old configuration waited three seconds to collect messages before processing. Its model loop commonly made several sequential calls with about 16,000 input tokens per call. Recorded simulation model-loop timings had a median of 9.34 seconds; those timings excluded the message wait and some preliminary work.

Current changes:

- Message batching defaults to 250 ms (150 ms for quick replies).
- Plain confirmations skip the separate remote intent-classification call.
- Simple catalog questions, greetings and supported shop facts use local handlers.
- Local option lists retain the exact displayed ordering for later selections.
- Complex questions and order changes retain the validated AI/tool workflow.

An isolated API measurement using five demo products and the existing configured main model returned a greeting in 0.188 seconds, a simple catalog query in 0.170 seconds, and a free-form birthday recommendation in 10.808 seconds. These are individual observations, not a latency guarantee or a load benchmark. A smaller-model comparison also exceeded three seconds.

A strict completed-answer deadline requires moving more conversation steps out of the remote model loop and defining local behavior for unsupported requests. An immediate acknowledgment is not the same as completing the requested action within that deadline.
