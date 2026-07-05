import json 
import torch 
import torch.nn as nn

from data import get_dataloader
import models

from fit import Trainer

SOURCE_DATA     = "orgs"                 
CKPT            = "orgs_pretrained.pt"   
PRETRAIN_EPOCHS = 18                     
LR              = 1e-3                    

def set_seed(seed=42):
    torch.manual_seed(seed)                                                                       
    torch.cuda.manual_seed_all(seed)

def main():
    with open("config.json", "r") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on:", device)

    train_loader, val_loader, _ = get_dataloader(
        data=SOURCE_DATA, data_path=config["DATA_PATH"], batch_size=config["BATCH_SIZE"])
    
    in_ch = train_loader.dataset.tensors[0].shape[1]
    n_cls = int(train_loader.dataset.tensors[1].max()) + 1
    print(f"Pretraining on {SOURCE_DATA} with {in_ch} input channels and {n_cls} classes.")

    #resnet 
    set_seed(42)  # Reset seed for reproducibility
    model = models.ResNet(in_channels=in_ch, num_classes=n_cls, drop_rate=config["DROP_RATE"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    trainer = Trainer(model, criterion, optimizer, device)
    trainer.fit(train_loader, val_loader, epochs=PRETRAIN_EPOCHS)

    # Save the pretrained model
    torch.save(model.state_dict(), CKPT)
    print(f"Pretrained model saved to {CKPT}")

if __name__ == "__main__":
    main()


