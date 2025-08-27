import os

full_path = os.path.realpath(__file__)

categories = []
with open(os.path.dirname(full_path) + '/cat.txt', 'r') as file:
    categories = file.read().splitlines()
    # categories = [cat.strip() for cat in categories if cat.strip()]

cat_array = ",".join(categories).split(",---,")
dic = {cat.split(":,")[0]:[s.title() for s in cat.split(":,")[1].split(",")]  for cat in cat_array}

# print(dic)
print(os.path.dirname(full_path))
for k in dic:
    print(k)
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(full_path)), "datasets", "multilabel_fashion_dataset", k), exist_ok=True)
    for cat in dic[k]:
        os.makedirs(os.path.join(os.path.dirname(os.path.dirname(full_path)), "datasets", "multilabel_fashion_dataset", k, cat), exist_ok=True)