import torch
import cv2

from cvae import Data_Manager,CVAE

# --------------------------------------------------------------------------------
if __name__ == '__main__':
    device_id = '0'

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(device_id))

    # Data Manager
    data_manager = Data_Manager()

    # C-VAE
    model = CVAE(data_manager, device, epochs=10)
    model.fit()

    # Single Sample
    model.load_model('cvae_state.pt')
    sample = model.sample(idx=5)

    cv2.imwrite('sample_test.png', sample)





