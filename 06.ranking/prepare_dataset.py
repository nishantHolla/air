from sklearn.datasets import fetch_20newsgroups
from sklearn.utils import Bunch
import numpy as np
from pathlib import Path
import sys
import json

selected_topics: list[str] = [
    "comp.graphics",
    "comp.os.ms-windows.misc",
    "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware",
    "comp.windows.x",
    "rec.autos",
    "rec.motorcycles",
    "rec.sport.baseball",
    "rec.sport.hockey",
    "sci.space",
]
docs_per_topic: int = 100

data = fetch_20newsgroups(
    subset="all",
    categories=selected_topics,
    shuffle=True,
    random_state=42,
    remove=("headers", "footers", "quotes"),
)
assert type(data) is Bunch


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <dataset_folder>")
        exit(1)

    elif not Path(sys.argv[1]).is_dir():
        print(f"{sys.argv[1]} is not a directory")
        exit(1)

    documents = []

    for topic_id, topic_name in enumerate(data.target_names):
        indices = np.where(data.target == topic_id)[0][:docs_per_topic]

        for i, idx in enumerate(indices):
            documents.append({"name": f"{topic_name}:{i}", "text": data.data[idx]})

    with open(Path(sys.argv[1]) / "20newsgroups.json", "w") as f:
        json.dump(documents, f, indent=4)
