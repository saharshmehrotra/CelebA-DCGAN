# **CelebA DCGAN: Generating Faces with Deep Convolutional GANs**

This project implements a **Deep Convolutional Generative Adversarial Network (DCGAN)** to generate realistic human face images using the **CelebA** dataset.

---

## **Dataset**
The model is trained on the **CelebA (CelebFaces Attributes) dataset**, which consists of over **200,000** celebrity face images.  
- The dataset is **automatically downloaded** if not already available.  
- Images are **center cropped** and resized to **64×64** pixels.

---

## **Project Structure**
📂 DCGAN-CelebA  
│── 📜 `train.py`              # Main script to train the DCGAN model  
│── 📜 `data_loader.py`        # Loads and preprocesses the CelebA dataset  
│── 📜 `model.py`              # Defines the Generator and Discriminator models  
│── 📂 `output/`               # Stores generated images & model checkpoints  
│── 📜 `README.md`             # Project documentation  
│── 📜 `requirements.txt`      # List of dependencies for easy installation  

---

## **Training the DCGAN**

Run the training script:
```bash
python train.py
```

By default:
- The dataset is stored in `./data/celeba/`
- Generated images and model checkpoints are saved in `./output/`

### **Custom Training Parameters**
You can modify `train.py` for different settings:

| Parameter       | Description                               | Default |
|-----------------|-------------------------------------------|---------|
| `num_epochs`    | Number of training epochs                 | 10      |
| `batch_size`    | Training batch size                       | 128     |
| `latent_dim`    | Noise vector size                         | 100     |
| `lr`            | Learning rate                             | 0.0002  |
| `save_interval` | Save model & images every N epochs        | 2       |

---

## **Generated Images**
After training, generated images and model checkpoints appear in the `./output/` directory, for example:

```text
📂 output/
│── fake_samples_epoch_2.png
│── fake_samples_epoch_4.png
│── netG_epoch_2.pth
│── netD_epoch_2.pth
```

To view a generated image:
```python
from PIL import Image
img = Image.open('output/fake_samples_epoch_10.png')
img.show()
```

---

## **Model Overview**

### Generator (`model.py`)
Converts a random noise vector (e.g., 100-dimensional) into a 64×64 RGB image using transpose convolutions, BatchNorm, and ReLU.

### Discriminator (`model.py`)
Classifies 64×64 images as real or fake using strided convolutions, LeakyReLU, and a Sigmoid output.

---

## **Training Loop**
The main training loop in `train.py`:
- Loads the CelebA dataset from `data_loader.py`.
- Trains the Discriminator to distinguish real vs. fake images.
- Trains the Generator to fool the Discriminator.
- Saves generated images and checkpoints at intervals.

---

## **Saving & Loading Models**
Model checkpoints are stored in `./output/`, for example:
```text
netG_epoch_10.pth  # Generator
netD_epoch_10.pth  # Discriminator
```

To load a trained Generator and generate new images:
```python
import torch
from model import Generator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
netG = Generator(latent_dim=100).to(device)
netG.load_state_dict(torch.load("output/netG_epoch_10.pth", map_location=device))
netG.eval()

# Generate a random image
noise = torch.randn(1, 100, 1, 1, device=device)
fake_image = netG(noise).detach().cpu()

import matplotlib.pyplot as plt
import numpy as np
plt.imshow(np.transpose(fake_image[0], (1, 2, 0)))
plt.axis("off")
plt.show()
```

---

## **Acknowledgments**
- PyTorch for its deep learning framework.
- CelebA Dataset creators for providing a comprehensive dataset of celebrity faces.

---

## **License**
This project is licensed under the MIT License.
