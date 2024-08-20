import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.transforms import ToTensor, Compose
from PIL import Image, ImageDraw
import numpy as np
import json
import cv2
from tqdm import tqdm
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
import matplotlib.pyplot as plt

# Training configurations
BATCH_SIZE = 1
LEARNING_RATE = 0.001
EPOCHS = 100
NUM_CLASSES = 33
ROOT_DIR = '/data2/Shaily/final/MRseg'
CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'earlystop/checkpoints/dental100')
RESULTS_DIR = os.path.join(ROOT_DIR, 'earlystop/resultsnew/dental100')
SAVE_INTERVAL = 5
PATIENCE = 10  # Early stopping patience
BEST_MODEL_DIR = os.path.join(ROOT_DIR, 'earlystop/best_models')
os.makedirs(BEST_MODEL_DIR, exist_ok=True)

# Dataset preparation
class TeethDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)

        self.images = {image['id']: image for image in self.coco_data['images']}
        self.annotations = self.coco_data['annotations']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_info = list(self.images.values())[index]
        image_path = os.path.join(self.root_dir, image_info['file_name'])
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        annotations = [ann for ann in self.annotations if ann['image_id'] == image_info['id']]

        boxes = []
        labels = []
        masks = []

        for ann in annotations:
            bbox = ann['bbox']
            boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
            labels.append(ann['category_id'])

            mask = Image.new('L', (image_info['width'], image_info['height']), 0)
            for segmentation in ann['segmentation']:
                polygon = np.array(segmentation).reshape(-1, 2)
                ImageDraw.Draw(mask).polygon([tuple(p) for p in polygon], outline=1, fill=1)
            masks.append(np.array(mask))

        if len(boxes) == 0:
            return None

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(np.array(masks, dtype=np.uint8))

        target = {'boxes': boxes, 'labels': labels, 'masks': masks}

        return image, target

# Data loading
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None
    return tuple(zip(*batch))

def get_teeth_data(train_root_dir, train_annotation_file, val_root_dir, val_annotation_file):
    transform = Compose([
        ToTensor()
    ])
    train_dataset = TeethDataset(train_root_dir, train_annotation_file, transform=transform)
    val_dataset = TeethDataset(val_root_dir, val_annotation_file, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader

# Model preparation
def get_instance_segmentation_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask,
                                                                                              hidden_layer, num_classes)
    return model


# Visualize composite masks
def visualize_composite_masks(dataset, num_samples=5):
    for i in range(num_samples):
        image, target = dataset[i]
        masks = target['masks'].numpy()
        boxes = target['boxes'].numpy()
        labels = target['labels'].numpy()
        image_np = image.permute(1, 2, 0).numpy()
        image_np = (image_np * 255).astype(np.uint8)

        # Ensure image is uint8
        if image_np.dtype != np.uint8:
            image_np = image_np.astype(np.uint8)

        # Convert to BGR if necessary (assuming RGB from PyTorch)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Ensure composite_mask has the same height and width as the image, and 3 channels
        composite_mask = np.zeros((image_np.shape[0], image_np.shape[1], 3), dtype=np.uint8)

        for j in range(masks.shape[0]):
            mask = masks[j].astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            color = tuple(np.random.randint(0, 255, size=3).tolist())
            composite_mask = cv2.drawContours(composite_mask, contours, -1, color, thickness=cv2.FILLED)
            cv2.rectangle(image_np, (int(boxes[j][0]), int(boxes[j][1])), (int(boxes[j][2]), int(boxes[j][3])), color, 2)
            cv2.putText(image_np, f"Label: {labels[j]}", (int(boxes[j][0]), int(boxes[j][1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Convert image_np back to RGB for plotting with matplotlib
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.addWeighted(image_np, 1, composite_mask, 0.5, 0))
        plt.title(f'Composite Mask for Image {i + 1}')
        plt.axis('off')
        plt.show()

# Training loop with gradient accumulation and early stopping
def train_one_epoch(model, optimizer, data_loader, device, epoch, accumulation_steps=4):
    model.train()
    progress_bar = tqdm(data_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]", unit="batch")
    accumulated_loss = 0.0
    running_loss = 0.0  # To keep track of the loss for progress bar updates

    for i, batch in enumerate(progress_bar, start=1):
        if batch is None:
            continue
        images, targets = batch
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        try:
            loss_dict = model(images, targets)
        except RuntimeError as e:
            print("RuntimeError occurred:")
            print("Images:", images)
            print("Targets:", targets)
            print("Error message:", str(e))
            raise e

        losses = sum(loss for loss in loss_dict.values())
        losses = losses / accumulation_steps

        # Ensure losses is a single scalar tensor
        losses = losses.mean()

        if not isinstance(losses, torch.Tensor) or losses.numel() != 1:
            raise ValueError(f"Losses should be a single-element tensor, got: {losses}")

        accumulated_loss += losses.item()
        running_loss += losses.item()
        losses.backward()

        if i % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Update progress bar every 10 batches
        if i % 10 == 0:
            progress_bar.set_postfix(loss=running_loss / 10)
            running_loss = 0.0

    print(f"Epoch [{epoch+1}] Training Loss: {accumulated_loss:.4f}")

def validate(model, data_loader, device, epoch):
    model.eval()  # Set the model to evaluation mode
    progress_bar = tqdm(data_loader, desc=f"Validating Epoch [{epoch+1}/{EPOCHS}]", unit="batch")
    val_loss = 0.0
    running_loss = 0.0  # To keep track of the loss for progress bar updates

    with torch.no_grad():  # Disable gradient computation for efficiency
        for i, batch in enumerate(progress_bar, start=1):
            if batch is None:
                continue
            images, targets = batch
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Model's forward pass in evaluation mode; only pass images and targets to calculate loss
            loss_dict = model(images, targets)


            # Save a few images with masks and bounding boxes drawn for visualization
            if i <= 5:  # Save only a few images per epoch for visualization
                outputs = model(images)  # Get outputs (predictions) from the model
                for j in range(len(outputs)):
                    image = images[j].permute(1, 2, 0).cpu().numpy()
                    image = (image * 255).astype(np.uint8)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                    output = outputs[j]
                    boxes = output['boxes'].detach().cpu().numpy().astype(int)
                    labels = output['labels'].detach().cpu().numpy()
                    scores = output['scores'].detach().cpu().numpy()
                    masks = output['masks'].detach().cpu().numpy()
                
                    for box, label, score, mask in zip(boxes, labels, scores, masks):
                        if score >= 0.4:  # Threshold for drawing
                            color = tuple(np.random.randint(0, 255, size=3).tolist())
                            mask = np.squeeze(mask).astype(np.uint8)
                            mask = (mask > 0.5).astype(np.uint8) * 255
                            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            image = cv2.drawContours(image, contours, -1, color, 2)
                            image = cv2.addWeighted(image, 0.7, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.3, 0)
                            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
                            cv2.putText(image, f" {label}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                    result_image_path = os.path.join(RESULTS_DIR, f"result_epoch_{epoch}_image_{i}_{j}.jpg")
                    cv2.imwrite(result_image_path, image)

    #print(f"Epoch [{epoch+1}] Validation Loss: {val_loss:.4f}")

    return val_loss



# Main function with early stopping and best model saving
def main():
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    train_root_dir = os.path.join(ROOT_DIR, 'Dental/train')
    val_root_dir = os.path.join(ROOT_DIR, 'Dental/valid')
    train_annotation_file = os.path.join(train_root_dir, 'trainutn.json')
    val_annotation_file = os.path.join(val_root_dir, 'validutn.json')

    train_loader, val_loader = get_teeth_data(train_root_dir, train_annotation_file, val_root_dir, val_annotation_file)

    # Visualize composite masks before training
    #visualize_composite_masks(TeethDataset(train_root_dir, train_annotation_file, transform=Compose([ToTensor()])))

    model = get_instance_segmentation_model(NUM_CLASSES)
    model = nn.DataParallel(model, device_ids=[2,7,5])  # Use DataParallel
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)  # L2 Regularization with weight_decay

    # Create checkpoint and results directories if they don't exist
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    best_val_loss = float('inf')
    patience_counter = 0
    best_dice_score = 0.0

    for epoch in range(EPOCHS):
        train_one_epoch(model, optimizer, train_loader, device, epoch)

        torch.cuda.empty_cache()

        val_loss = validate(model, val_loader, device, epoch)

        # Early stopping and best model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        
            best_model_path = os.path.join(BEST_MODEL_DIR, f"best_model_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, best_model_path)
        else:
            patience_counter += 1
        
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            
            # Save the model at the point of early stopping
            early_stop_checkpoint_path = os.path.join(CHECKPOINT_DIR, f"early_stop_checkpoint_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),  # Save the model state dict
                'optimizer_state_dict': optimizer.state_dict()
            }, early_stop_checkpoint_path)
            
            break


        # Save checkpoint every SAVE_INTERVAL epochs
        if (epoch + 1) % SAVE_INTERVAL == 0 or epoch == EPOCHS - 1:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),  # Save the model state dict
                'optimizer_state_dict': optimizer.state_dict()
            }, checkpoint_path)

    print("Training completed!")

if __name__ == '__main__':
    main()
