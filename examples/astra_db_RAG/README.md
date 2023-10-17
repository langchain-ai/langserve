# Astra DB and LangServe

A simple demo of integrating Astra DB in LangServe.

Workloads demonstrated: vector-powered RAG, LLm caching.

### Setup

Make sure you have an OpenAI API Key and an [Astra DB](https://astra.datastax.com) instance with Vector.
For the latter, get a "Database Administrator" [token](https://awesome-astra.github.io/docs/pages/astra/create-token/#c-procedure) and copy your [Database ID](https://awesome-astra.github.io/docs/pages/astra/faq/#where-should-i-find-a-database-identifier).

Copy `cp .env.template .env` and fill it with the above secrets/connection parameters.

Source the file `source .env`

Ensure you have the requirements: `pip install -r requirements.txt`.

Launch the script to create and populate the vector store (just once): `python setup.py`. This inserts a handful of sample texts in the vector store from a local file.

Now you can start the LangServe-powered API: `python server.py`.

> If you want to clear the LLM cache for demonstration purposes: `python server.py clear_cache`.

### Test

The API offers two sample routes.

**Route 1**: a _synonym finder_ demonstrates LLM caching with Astra DB.

**Route 2**: an _entomology chain_ demonstrates RAG using Astra DB as the vector store (while also benefitting from the LLM caching as well).

_Suggestion: to see caching at work, try repeating the same invocation and check the response time._

#### Test (script)

Launch `python client.py` (which includes using the streaming form of the exposed RAG chain).

#### Test (curl)

Try these:

```
curl -XPOST \
    localhost:8000/synonyms/invoke \
    -H 'Content-Type: application/json' \
    -d '{"input":{"word":"crawl"}}'
```

```
curl -XPOST \
    localhost:8000/entomology/invoke \
    -H 'Content-Type: application/json' \
    -d '{"input":"Are there more coleoptera or bugs?"}'
```

#### Test (OpenAPI)

Point your browser to `http://localhost:8000/docs`, pick the `*/invoke` endpoints, hit the "Try it out" button and then "Execute" after playing with the payload JSON.
