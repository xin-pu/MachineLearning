from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
input_classes = ['x', 'y', 'a', 'b', 'c']
label_encoder.fit(input_classes)


def printTest():
    for item in label_encoder.classes_:
        print(item)

    for item in enumerate(label_encoder.classes_):
        print("{0}_{1}".format(item[0], item[1]))


labels = ['a', 'x']
encoded_labels = label_encoder.transform(labels)
print(encoded_labels)

encoded_labels = [0, 1, 2, 3, 4]
labels = label_encoder.inverse_transform(encoded_labels)
print(labels)
