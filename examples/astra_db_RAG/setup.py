import os

import cassio

from langchain.vectorstores import Cassandra
from langchain.embeddings import OpenAIEmbeddings

# DB init
cassio.init(
    token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
    database_id=os.environ["ASTRA_DB_ID"],
    keyspace=os.environ.get("ASTRA_DB_KEYSPACE"),
)


if __name__ == '__main__':
    embeddings = OpenAIEmbeddings()
    vector_store = Cassandra(
        session=None,
        keyspace=None,
        embedding=embeddings,
        table_name="langserve_demo_store",
    )
    #
    lines = [
        l.strip()
        for l in open("sources.txt").readlines()
        if l.strip()
        if l[0] != "#"
    ]
    ids = [
        "_".join(l.split(" ")[:2]).lower().replace(":", "")
        for l in lines
    ]
    #
    vector_store.add_texts(texts=lines, ids=ids)
    print(f"Done ({len(lines)}).")