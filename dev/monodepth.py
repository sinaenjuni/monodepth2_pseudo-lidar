import os
import sys
import torch
from torchvision import transforms

import tools
import networks
from utils import download_model_if_doesnt_exist

class Monodepth():
    def __init__(self, args):
        assert args.model_name is not None, \
            "You must specify the --model_name parameter; see README.md for an example"

        if torch.cuda.is_available() and args.cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print("Deivce is {}".format(self.device))

        download_model_if_doesnt_exist(args.model_name)
        self.model_path = os.path.join("models", args.model_name)
        print("-> Loading model from ", self.model_path)
        self.encoder_path = os.path.join(self.model_path, "encoder.pth")
        self.depth_decoder_path = os.path.join(self.model_path, "depth.pth")

        # LOADING PRETRAINED MODEL
        print("   Loading pretrained encoder")
        self.encoder = networks.ResnetEncoder(18, False)
        loaded_dict_enc = torch.load(self.encoder_path, map_location=self.device)

        # extract the height and width of image that this model was trained with
        self.feed_height = loaded_dict_enc['height']
        self.feed_width = loaded_dict_enc['width']
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(filtered_dict_enc)
        self.encoder.to(self.device)
        self.encoder.eval()

        print("   Loading pretrained decoder")
        self.depth_decoder = networks.DepthDecoder(
            num_ch_enc=self.encoder.num_ch_enc, scales=range(4))

        loaded_dict = torch.load(self.depth_decoder_path, map_location=self.device)
        self.depth_decoder.load_state_dict(loaded_dict)

        self.depth_decoder.to(self.device)
        self.depth_decoder.eval()

    def get_disparity_map(self, image_path:str):
        image = tools.load_image(image_path=image_path)
        original_width, original_height = image.size
        resized_image = tools.resize_image(image=image, tsize=(self.feed_width, self.feed_height))
        input_image = transforms.ToTensor()(resized_image).unsqueeze(0)

        # PREDICTION
        with torch.no_grad():
            input_image = input_image.to(self.device)
            features = self.encoder(input_image)
            outputs = self.depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(disp, 
                                                        (original_height, original_width), 
                                                        mode="bilinear", 
                                                        align_corners=False)
        return disp.cpu().numpy().squeeze(), disp_resized.cpu().numpy().squeeze()
