from src.vectordb import create_index, insert_data,  drop_collection, connect, create_collection
import numpy as np

connect()

collection_name = 'test'
drop_collection(collection_name)
collection = create_collection(collection_name)
insert_data(collection)
create_index(collection)

search_params = {
    "metric_type": "L2",
    "params": {"nprobe": 10},
}
rng = np.random.default_rng(seed=42)
vectors_to_search = rng.random((1, 8))

collection.load()
result = collection.search(vectors_to_search, "embeddings", search_params, limit=3, expr="random > 0.5", output_fields=["random"])

drop_collection(collection_name)
