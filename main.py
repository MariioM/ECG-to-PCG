from src.utils import plot_single_pair, seed_everything
from src.config import Config
from E2P.src.train_vae import train_vae

def main():
    seed_everything(Config.SEED)
    
    plot_single_pair()

    train_vae()

if __name__ == "__main__":
    main()