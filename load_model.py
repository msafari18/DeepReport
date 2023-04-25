from DeepReport_model.transformer import Transformer
import torch
from DeepReport_model.encoding import ResNetAE
from DeepReport_model.eval import eval_transformer
from skimage.transform import resize
import torchxrayvision as xrv


config = {
    "max_step": 200,
    "batch_size": 64,
    "lr": 1e-4,
    "n_epochs": 25,
    "alpha_c": 1.0,
    "grad_clip": 5.,
    "embed_dim": 300,
    "nheads": 8
}

def encode_input(input_image):
    feature_model = ResNetAE(weights='101-elastic')
    feature_model.eval()
    x = torch.randn(2, 1, 224, 224)
    assert feature_model.encode(x).shape == (2, 1024, 14, 14)

    with torch.no_grad():
        # id = sample['identifier']
        img = torch.from_numpy(input_image).unsqueeze(0)
        inputs = img
        feats = feature_model.encode(inputs).cpu().numpy()

    return feats

def load_transformer(input, vocab):


    img = xrv.datasets.normalize(input, 255)
    img = resize(img, (224, 224))
    print("image loaded")
    if len(img.shape) > 2:
        img = img[:, :, 0]
    if len(img.shape) < 2:
        print("error, dimension lower than 2 for image")
    img = img[None, :, :]

    feats = encode_input(img)
    print("features loaded")
    vocab_size = len(vocab.idx2word)
    encoder_layers, decoder_layers = 8, 8
    device = "cpu"
    model = Transformer(vocab_size=vocab_size, embed_dim=config["embed_dim"],
                        encoder_layers=encoder_layers, decoder_layers=decoder_layers, n_heads=config["nheads"],
                        max_len=config["max_step"], device=device).to(device)

    model.load_state_dict(torch.load("DeepReport_model/transformer_decoder_8heads_maxword200.pth", map_location=torch.device(device)))
    model.eval()
    report = eval_transformer(model, device, vocab, feats, max_step=200, beam_size=3)

    return report

