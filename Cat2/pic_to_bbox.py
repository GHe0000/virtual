import numpy as np
import psd_tools

psd_path = 'cat4.psd'
psd = psd_tools.PSDImage.open(psd_path)
# print(psd)
for l in psd:
	print(l.name)
	print(np.array(l.bbox))
