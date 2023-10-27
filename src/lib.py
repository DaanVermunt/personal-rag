from src.embedding import get_embeddings, embed_str
from src.vectordb import connect, create_collection, insert_data, create_index, my_collection_name, get_collection


def init_db():
    connect()

    collection = create_collection(my_collection_name)
    embeddings, lines = get_embeddings()
    insert_data(collection, embeddings, lines)
    create_index(collection)
    return collection


def search_str(inp: str):
    connect()

    collection = get_collection(my_collection_name)
    collection.load()

    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10},
    }
    vectors_to_search = [embed_str(inp)]

    result = collection.search(vectors_to_search, "embeddings", search_params, limit=3, output_fields=["text"])
    return result
