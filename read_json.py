import json
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

with open("mosaic_params.json") as f:
    data = json.load(f)

print(f"The images are {data['Input files']}")
corrs = data["Correspondences"]
print(f"There are {len(corrs)} sets of correspondences")
for i in range(len(corrs)):
    print(f"There are {len(corrs[i][0][1])} correspdondences between image {corrs[i][0][0]} and image {corrs[i][1][0]}")
    corra = np.asarray(corrs[i][0][1])
    corrb = np.asarray(corrs[i][1][1])   
    print(corra.T)
    print(corrb.T)
    imageNamea = corrs[i][0][0]
    imageNameb = corrs[i][1][0]
    imageNamea = imageNamea.split('.')
    imageNameb = imageNameb.split('.')
    imagea = io.imread(imageNamea[0]+'.png')
    imageb = io.imread(imageNameb[0]+'.png')
    fig, ax = plt.subplots(1,2)
    fig.set_size_inches(10.5,10.5)
    ax[0].imshow(imagea)
    ax[0].scatter(corra[:,0],corra[:,1],c='#ff7f0e')
    ax[1].imshow(imageb)
    ax[1].scatter(corrb[:,0],corrb[:,1],c='#ff7f0e')
    plt.savefig(str(i)+'.png')
    print(corra.shape)
    print(corrb.shape)
print(f"The output file is {data['Output file']}")
