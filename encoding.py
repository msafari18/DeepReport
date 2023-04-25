import os
import h5py
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torchxrayvision as xrv
from torchxrayvision.datasets import *
from tqdm import tqdm
import torchxrayvision as xrv
from torchxrayvision.autoencoders import *
from torchxrayvision.autoencoders import _ResNetAE


class MyResNet101(_ResNetAE):
    def encode(self, x, check_resolution=True):

        if check_resolution and hasattr(self, 'weights_metadata'):
            resolution = self.weights_metadata['resolution']
            if (x.shape[2] != resolution) | (x.shape[3] != resolution):
                raise ValueError("Input size ({}x{}) is not the native resolution ({}x{}) for this model. Set check_resolution=False on the encode function to override this error.".format(
                    x.shape[2], x.shape[3], resolution, resolution))

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


def ResNetAE101(**kwargs):
    return MyResNet101(Bottleneck, DeconvBottleneck, [3, 4, 23, 2], 1, **kwargs)


def ResNetAE(weights=None):

    if weights == None:
        return ResNetAE101()

    if not weights in model_urls.keys():
        raise Exception("weights value must be in {}".format(
            list(model_urls.keys())))

    method_to_call = globals()[model_urls[weights]["class"]]
    ae = method_to_call()

    # load pretrained models
    url = model_urls[weights]["weights_url"]
    weights_filename = os.path.basename(url)
    weights_storage_folder = os.path.expanduser(
        os.path.join("~", ".torchxrayvision", "models_data"))
    weights_filename_local = os.path.expanduser(
        os.path.join(weights_storage_folder, weights_filename))

    if not os.path.isfile(weights_filename_local):
        print("Downloading weights...")
        print("If this fails you can run `wget {} -O {}`".format(url,
              weights_filename_local))
        pathlib.Path(weights_storage_folder).mkdir(parents=True, exist_ok=True)
        download(url, weights_filename_local)

    try:
        state_dict = torch.load(weights_filename_local, map_location='cpu')
        ae.load_state_dict(state_dict)
    except Exception as e:
        print("Loading failure. Check weights file:", weights_filename_local)
        raise (e)

    ae = ae.eval()

    ae.weights = weights
    ae.weights_metadata = model_urls[weights]
    ae.description = model_urls[weights]["description"]

    return ae


class My_MIMIC_Dataset(MIMIC_Dataset):
    def __init__(self,
                 imgpath,
                 csvpath,
                 metacsvpath,
                 views=["PA"],
                 transform=None,
                 data_aug=None,
                 flat_dir=True,
                 seed=0,
                 unique_patients=True
                 ):
        super(My_MIMIC_Dataset, self).__init__(imgpath,
                                               csvpath,
                                               metacsvpath,
                                               views,
                                               transform,
                                               data_aug,
                                               flat_dir,
                                               seed,
                                               unique_patients)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        subjectid = str(self.csv.iloc[idx]["subject_id"])
        studyid = str(self.csv.iloc[idx]["study_id"])
        dicom_id = str(self.csv.iloc[idx]["dicom_id"])

        identitfier = "p" + subjectid[:2] + '/' "p" + \
            subjectid + '/' "s" + studyid + '/' + dicom_id
        img_path = os.path.join(self.imgpath, identitfier + ".jpg")
        img = imread(img_path)

        sample["img"] = normalize(img, maxval=255, reshape=True)

        sample["identifier"] = identitfier

        sample = apply_transforms(sample, self.transform)

        return sample