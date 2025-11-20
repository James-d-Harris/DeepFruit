from pre_processing import preprocess_image

features = preprocess_image("fruits-360_100x100/fruits-360/Training/Apple 5/r0_0_100.jpg", plot=True)
x = features["norm"]
