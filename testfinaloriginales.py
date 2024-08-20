import torch
import os
import torchvision
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Compose
from PIL import Image, ImageDraw
import cv2
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
import json
from torchvision.ops import nms
from sklearn.metrics import jaccard_score, precision_score, recall_score, accuracy_score, f1_score

# Configurations
ROOT_DIR = '/data2/Shaily/final/MRseg'
CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'earlystop/checkpoints/dental100')
TEST_DIR = os.path.join(ROOT_DIR, 'Dental/test')
TEST_ANNOTATION_FILE = os.path.join(TEST_DIR, 'testutn.json')
RESULTS_DIR = os.path.join(ROOT_DIR, 'earlystop/test_results')
NUM_CLASSES = 33
BATCH_SIZE = 1
CONFIDENCE_THRESHOLD = 0.4
IOU_THRESHOLD = 0.5
IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05)

colors = {
    'Background': [0, 0, 0],
    '1': [127, 191, 127],   # Lighter Green
    '2': [191, 127, 127],   # Lighter Red
    '3': [127, 127, 191],   # Lighter Blue
    '4': [191, 191, 127],   # Lighter Olive
    '5': [127, 191, 191],   # Lighter Teal
    '6': [191, 127, 191],   # Lighter Purple
    '7': [255, 127, 127],   # Lighter Bright Red
    '8': [127, 255, 127],   # Lighter Bright Green
    '9': [127, 127, 255],   # Lighter Bright Blue
    '10': [255, 255, 127],   # Lighter Yellow
    '11': [127, 255, 255],   # Lighter Cyan
    '12': [255, 127, 255],   # Lighter Magenta
    '13': [223, 223, 223],   # Lighter Silver
    '14': [163, 158, 197],   # Lighter Dark Slate Blue
    '15': [191, 191, 255],   # Lighter Light Blue
    '16': [255, 191, 191],   # Lighter Light Red
    '17': [191, 255, 191],   # Lighter Light Green
    '18': [255, 191, 255],   # Lighter Light Magenta
    '19': [191, 255, 255],   # Lighter Light Cyan
    '20': [255, 255, 191],   # Lighter Light Yellow
    '21': [255, 210, 127],   # Lighter Orange
    '22': [255, 162, 127],   # Lighter Red-Orange
    '23': [247, 192, 247],   # Lighter Violet
    '24': [165, 127, 192],   # Lighter Indigo
    '25': [247, 242, 197],   # Lighter Khaki
    '27': [232, 180, 142],   # Lighter Chocolate
    '28': [250, 209, 175],   # Lighter Sandy Brown
    '29': [143, 216, 213],   # Lighter Light Sea Green
    '30': [255, 236, 220],   # Lighter Peach Puff
    '31': [189, 180, 246],   # Lighter Medium Slate Blue
    '32': [255, 235, 127]    # Lighter Gold
}
os.makedirs(RESULTS_DIR, exist_ok=True)


# Dataset preparation (reuse from your training code)
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

        target = {'boxes': boxes, 'labels': labels, 'masks': masks, 'image_id': image_info['id'], 'file_name': image_info['file_name']}

        return image, target

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None
    return tuple(zip(*batch))

def get_test_data(test_root_dir, test_annotation_file):
    transform = Compose([
        ToTensor()
    ])
    test_dataset = TeethDataset(test_root_dir, test_annotation_file, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    return test_loader

# Load the best model
def load_model(num_classes, device, checkpoint_path):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask,
                                                                                              hidden_layer, num_classes)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Adjust the state_dict by removing "module." if necessary
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # remove "module." prefix
        else:
            new_state_dict[k] = v

    # Load the modified state_dict into the model
    model.load_state_dict(new_state_dict)

    model.to(device)
    model.eval()
    return model

def get_instance_color(label):
    """Get color for a given label."""
    return colors.get(str(label), [255, 255, 255])  # Default to white if label not found


def calculate_metrics(pred_mask, true_mask):
    pred_mask = pred_mask.flatten()
    true_mask = true_mask.flatten()
    
    # Calculate various metrics
    dice = (2. * np.sum(pred_mask * true_mask)) / (np.sum(pred_mask) + np.sum(true_mask) + 1e-5)
    jaccard = jaccard_score(true_mask, pred_mask)
    precision = precision_score(true_mask, pred_mask)
    recall = recall_score(true_mask, pred_mask)
    accuracy = accuracy_score(true_mask, pred_mask)
    specificity = np.sum((true_mask == 0) & (pred_mask == 0)) / np.sum(true_mask == 0)
    
    return dice, accuracy, jaccard, precision, recall, specificity

def calculate_average_metrics(all_metrics):
    avg_metrics = {key: np.mean([m[key] for m in all_metrics if key in m]) for key in all_metrics[0]}
    return avg_metrics
    

# Testing function
def test(model, data_loader, device, show_first_n_results=3):
    model.eval()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    metrics = {
        'dice': [],
        'jaccard': [],
        'precision': [],
        'recall': [],
        'accuracy': [],
        'specificity': []
    }

    all_image_metrics =[]
    
    with torch.no_grad():
        for idx, (images, targets) in enumerate(tqdm(data_loader, desc="Testing")):
            if images is None or targets is None:
                continue
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]

            outputs = model(images)

            for i in range(len(outputs)):
                image = images[i].permute(1, 2, 0).cpu().numpy()
                image = (image * 255).astype(np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                output = outputs[i]
                boxes = output['boxes'].detach().cpu().numpy().astype(int)
                labels = output['labels'].detach().cpu().numpy()
                scores = output['scores'].detach().cpu().numpy()
                masks = output['masks'].detach().cpu().numpy()

                # Apply confidence score threshold
                keep = scores >= CONFIDENCE_THRESHOLD
                boxes = boxes[keep]
                labels = labels[keep]
                scores = scores[keep]
                masks = masks[keep]

                # Apply Non-Maximum Suppression (NMS)
                keep = nms(torch.tensor(boxes, dtype=torch.float32), torch.tensor(scores), IOU_THRESHOLD)
                boxes = boxes[keep]
                labels = labels[keep]
                masks = masks[keep]

                # Collect true labels and masks for metrics calculation
                true_boxes = targets[i]['boxes'].cpu().numpy().astype(int)
                true_labels = targets[i]['labels'].cpu().numpy()
                true_masks = targets[i]['masks'].cpu().numpy()

                # Ensure binary masks for Dice calculation
                masks = (masks > 0.5).astype(np.uint8)
                true_masks = (true_masks > 0.5).astype(np.uint8)

                for true_label in np.unique(true_labels):
                    true_label_indices = np.where(true_labels == true_label)
                    true_label_boxes = true_boxes[true_label_indices]
                    true_label_masks = true_masks[true_label_indices]

                    pred_label_indices = np.where(labels == true_label)
                    pred_label_boxes = boxes[pred_label_indices]
                    pred_label_masks = masks[pred_label_indices]

                    image_metrics = []

                    for true_mask, pred_mask in zip(true_label_masks, pred_label_masks):
                        dice, accuracy, jaccard, precision, recall, specificity = calculate_metrics(pred_mask,
                                                                                                    true_mask)
                        if dice>0.5:
                            image_metrics.append({
                                'dice': dice,
                                'accuracy': accuracy,
                                'jaccard': jaccard,
                                'precision': precision,
                                'recall': recall,
                                'specificity': specificity
                            })

                if image_metrics:
                    avg_image_metrics = calculate_average_metrics(image_metrics)
                    all_image_metrics.append(avg_image_metrics)
                    print(avg_image_metrics)


                file_name = targets[i]['file_name']  # Get the original file name
                image_id = file_name.split('.')[0]
                os.makedirs(os.path.join(RESULTS_DIR, "pred"), exist_ok=True)
              

                
                # Draw predicted labels and instance segmentation masks
                for box, label, score, mask in zip(boxes, labels, scores, masks):
                    if score >= CONFIDENCE_THRESHOLD:
                        color = get_instance_color(label)
                
                        # Threshold and convert mask to single channel
                        mask = (mask > 0.5).astype(np.uint8)
                        mask = mask.squeeze()  # Ensure mask is single-channel

                        # Create a colored mask with the same color as the polygon
                        overlay = image.copy()
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for cnt in contours:
                            cv2.drawContours(overlay, [cnt], -1, color, -1)  # Fill polygon with color on overlay
                
                        # Blend the overlay with the original image
                        alpha = 0.25  # Transparency factor
                        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)  # Apply the transparent overlay

                        # Choose label color based on background brightness
                        label_color = (255, 255, 255)  # if np.mean(color) < 128 else (0, 0, 0)

                        # Draw label with contrasting color
                        cv2.putText(image, f"{label}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2)
                        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)  # Draw bounding box

                result_image_path = os.path.join(RESULTS_DIR,"pred", f"{image_id}_pred.jpg")
                cv2.imwrite(result_image_path, image)

    
    if all_image_metrics:
        avg_metrics = calculate_average_metrics(all_image_metrics)
        print("Average metrics for all images:", avg_metrics)
        
        # Save the metrics to a JSON file
        metrics_file_path = os.path.join(RESULTS_DIR, "metrics.json")
        with open(metrics_file_path, "w") as f:
            json.dump(avg_metrics, f, indent=4)
               

def main():
    device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')

    test_loader = get_test_data(TEST_DIR, TEST_ANNOTATION_FILE)

    # Load the best model
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "early_stop_checkpoint_epoch_10.pt")  # Replace xx with the appropriate epoch number
    model = load_model(NUM_CLASSES, device, checkpoint_path)

    # Run the test function
    test(model, test_loader, device)

if __name__ == '__main__':
    main()
