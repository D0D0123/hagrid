import json
import os

from pprint import pprint

for gesture in ["call", "peace", "peace_inverted", "like", "dislike", "ok", "mute", "stop", 
"stop_inverted", "fist"]:
    print(f'current gesture: {gesture}')
    f = open(f"annotations1/{gesture}.json", "r")

    test_imgnames = os.listdir(f'subsamples_test1/{gesture}/')
    train_imgnames = os.listdir(f'subsamples_train1/{gesture}/')

    data = json.load(f)

    out_test_f = open(f"annotations_test1/{gesture}.json", "w")
    out_train_f = open(f"annotations_train1/{gesture}.json", "w")

    out_test_dict = {}
    for imgname in sorted(test_imgnames):
        name = imgname[:-4]
        out_test_dict[name] = data[name]

    out_train_dict = {}
    for imgname in sorted(train_imgnames):
        name = imgname[:-4]
        out_train_dict[name] = data[name]

    json.dump(out_test_dict, out_test_f, indent=4)
    json.dump(out_train_dict, out_train_f, indent=4)

    f.close()
    out_test_f.close()
    out_train_f.close()


# print(test_imgnames)
# print(train_imgnames)