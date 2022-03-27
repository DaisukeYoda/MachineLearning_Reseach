import sys,os
sys.path.append(os.pardir)
from datasets.mnist import load_mnist
import numpy as np
from PIL import Image

(x_train, t_train), (x_test,t_test) = load_mnist(flatten=True,normalize=False)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size,batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

img = x_train[0]
label = t_train[0]

img = img.reshape(28,28)
pil_img = Image.fromarray(np.uint8(img))
pil_img.show()


