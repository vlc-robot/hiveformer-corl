import json
import random
from pathlib import Path
from PIL import Image
import streamlit as st
import pandas as pd
import numpy as np

st.title("Annotations tool instructions")
task = "tower3"
gif_dir = Path("datasets")


# load existing annotations
annotations_file = Path("annotations.json")
with open(annotations_file) as fid:
    annotations = json.load(fid)

# load task variations
var_file = Path(f"{task}.json")
with open(var_file) as fid:
    variations = json.load(fid)

# load examples
examples_file = "amt-train.fixed.json"
with open(examples_file) as fid:
    examples = json.load(fid)

# display 10 random examples
examples = random.choices(examples, k=5)
for example in examples:
    st.text(example["sentence"])

# prepare annotations json
done = set([])
for item in annotations:
    if not item["id"].startswith(f"streamlit-{task}"):
        continue
    if item["fields"]["task"] == task and item["fields"]["instruction"] != "":
        done.add(item["fields"]["variation"])

todo = []
for i, variation in enumerate(variations):
    if i < 100:
        continue
    if i in done:
        continue
    colors = [v[0] for v in variation]
    annotations.append(
        {
            "id": f"streamlit-{task}-{len(annotations)}",
            "createdTime": "2020-01-01T00:00:00.000Z",
            "fields": {
                "task": task,
                "variation": i,
                "colors": colors,
                "episode": 0,
                "gif_url": "",
                "instruction": "",
                "time_start": "",
                "Last Modified": "",
            },
        }
    )
    todo.append(len(annotations) - 1)

#    "id": "rec01TRmhtjjIITUk",
#    "createdTime": "2022-05-29T13:58:00.000Z",
#    "fields": {
#      "task": "push_buttons3",
#      "variation": 170,
#      "episode": 6,
#      "gif_url": "https://rlbench-annotations.guhur.workers.dev/push_buttons3-170-6-front.gif",
#      "instruction": "Press the grey and then the green button ",
#      "time_start": "1653837442221",
#      "Last Modified": "2022-05-29T15:18:08.000Z"
#    }
#  },

index = random.choice(todo)
item = annotations[index]
colors = item["fields"]["colors"]

print(index, item)
st.text(",".join(colors))

# image = Image.open(gif_dir/ f"variation{variation}.gif")
# st.image(image)

instr = st.text_area("Instruction")
print(instr)
with open("annotations.json", "w") as fid:
    json.dump(annotations, fid, indent=2)

st.text("Success!")

st.text("Press r for next item")
