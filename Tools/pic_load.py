import numpy as np
import matplotlib.pyplot as plt
import yaml
# img = np.load("mouthdata.npy")

# plt.imshow(img)
# plt.show()


yaml_path = "test_deep_inf.yaml"
with open(yaml_path, encoding='utf8') as f:
	inf = yaml.safe_load(f)
	a = inf['mouth']['path'][]
	print(a)
	img = np.load(a)
	plt.imshow(img)
	plt.show()