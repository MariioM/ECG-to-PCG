import torch
from src.config import Config
from src.models import PCGVAE
from src.dataset import get_dataloaders
from src.utils import visualize_vae_reconstruction

def main():
    cfg = Config.load_model_config()
    _, _, test_loader = get_dataloaders(Config)
    
    model = PCGVAE(
        in_channels=cfg['vae']['in_channels'],
        base_channels=cfg['vae']['base_channels'],
        z_dim=cfg['vae']['z_dim']
    ).to(Config.DEVICE)
    
    path_weights = Config.CHECK_DIR / cfg['paths']['vae_checkpoint']
    
    if path_weights.exists():
        model.load_state_dict(torch.load(path_weights, map_location=Config.DEVICE))
        print(f" Pesos cargados desde {path_weights}")

        visualize_vae_reconstruction(model, test_loader, Config.DEVICE)
    else:
        print(f" Error: No se encontró el archivo de pesos en {path_weights}")

if __name__ == "__main__":
    main()