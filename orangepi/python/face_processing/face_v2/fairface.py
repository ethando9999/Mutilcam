import os
import torch
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
import cv2
# from .mdp_aligner import FaceDetection
# from .dlib_aligner import FaceAligner

MODEL_4 = "res34_fair_align_multi_4_20190809.pt"
MODEL_7 = "res34_fair_align_multi_7_20190809.pt"
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models"
)

class FairFacePredictor:
    """
    FairFace predictor supporting two modes:
      - model_selection=0: 7-class (race+gender+age)
      - model_selection=1: 4-class (race only)
    Adds confidence thresholds for race, gender, and age.
    """
    def __init__(self, model_dir=MODEL_PATH, model_selection: int = 1,
                 r_conf: float = 0.3, g_conf: float = 0.5, a_conf: float = 0.5):
        self.model_dir = model_dir
        self.device = torch.device("cpu")
        self.model_selection = model_selection 
        # Confidence thresholds
        self.r_conf = r_conf
        self.g_conf = g_conf
        self.a_conf = a_conf

        # Preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Label mappings
        self.race_labels_7 = ['White', 'Black', 'Latino_Hispanic', 'East Asian',
                              'Southeast Asian', 'Indian', 'Middle Eastern']
        self.race_labels_7_merged = ['White', 'Black', 'Latino_Hispanic', 'Asian', 'Indian', 'Middle Eastern']
        self.race_labels_4 = ['White', 'Black', 'Asian', 'Indian']
        self.gender_labels = ['Male', 'Female']
        self.age_labels = ['0-2', '3-9', '10-19', '20-29', '30-39',
                           '40-49', '50-59', '60-69', '70+']

        # Load appropriate model
        model_file = MODEL_7 if model_selection == 0 else MODEL_4
        num_classes = 18
        self.model = self._load_resnet(os.path.join(model_dir, model_file), num_classes)


    def _load_resnet(self, model_path, num_classes):
        model = torchvision.models.resnet34(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    @staticmethod
    def _softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    
    def merge_race_logits_7_to_6(self, orig_logits: np.ndarray) -> np.ndarray:
        """
        Gộp logits của 7 nhãn race gốc thành 6 nhãn bằng cách:
        - Gộp 'East Asian' (index 3) và 'Southeast Asian' (index 4) thành 'Asian'
        - Sử dụng log-sum-exp để bảo toàn ý nghĩa xác suất sau softmax

        Args:
            orig_logits (np.ndarray): Mảng logits đầu vào với 7 phần tử

        Returns:
            np.ndarray: Mảng logits sau khi gộp còn 6 phần tử
        """
        asian_logit = np.logaddexp(orig_logits[3], orig_logits[4])
        merged_logits = np.array([
            orig_logits[0],  # White
            orig_logits[1],  # Black
            orig_logits[2],  # Latino_Hispanic
            asian_logit,     # Asian (gộp)
            orig_logits[5],  # Indian
            orig_logits[6],  # Middle Eastern
        ])
        return merged_logits

    def predict(self, image_np):
        """
        Predict on an RGB image np.ndarray of a face.
        Returns dict depending on model_selection:
          - 0: {'race':(label,conf), 'gender':..., 'age':...}
          - 1: {'race':(label,conf), 'race_probs_4':[...]}  
        If confidence below threshold, label is None.
        """
        # Preprocess
        tensor = self.transform(image_np).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(tensor).cpu().numpy().squeeze()

        if self.model_selection == 0:
            race_orig_logits = outputs[:7]
            merged_logits    = self.merge_race_logits_7_to_6(race_orig_logits)
            race_probs       = self._softmax(merged_logits)   # 6 class

            gender_probs = self._softmax(outputs[7:9])
            age_probs    = self._softmax(outputs[9:18])

            r_max = float(np.max(race_probs))
            g_max = float(np.max(gender_probs))
            a_max = float(np.max(age_probs))

            race_label   = self.race_labels_7_merged[int(np.argmax(race_probs))] if r_max >= self.r_conf else None
            gender_label = self.gender_labels[int(np.argmax(gender_probs))]      if g_max >= self.g_conf else None
            age_label    = self.age_labels[int(np.argmax(age_probs))]            if a_max >= self.a_conf else None

            return {
                'race':   (race_label,   r_max),
                'gender': (gender_label, g_max),
                'age':    (age_label,    a_max)
            }

        else:
            race_probs = self._softmax(outputs[:4])
            r_max = float(np.max(race_probs))
            race_label = self.race_labels_4[int(np.argmax(race_probs))] if r_max >= self.r_conf else None
            return {
                'race': (race_label, r_max),
                'race_probs_4': race_probs.tolist()
            }

def evaluate_model_7(
    csv_path: str = "data/fairface_label_val.csv",
    out_csv_name: str = "fairface_results_7.csv",
    show_classification_report: bool = True,
):
    """
    Evaluate 7-class FairFace model on validation set for race, gender, and age.

    Steps:
    1. Load CSV with 'file', 'race', 'gender', 'age'.
    2. Normalize ground truth: replace '_'->' ' for race, capitalize gender.
    3. Detect & align face, predict all 3 attributes.
    4. Normalize predictions and skip invalid.
    5. Compute accuracy, optional classification reports, confusion matrices.
    6. Save full results (including None preds) to output CSV.
    """
    import os
    from pathlib import Path
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        classification_report,
    )
    from tqdm import tqdm
    from mdp_aligner import FaceDetection

    # 1. Load and prepare ground truth
    df = pd.read_csv(csv_path)
    df['gt_race'] = df['race']          # e.g., 'Latino_Hispanic'
    df['gt_gender'] = df['gender'].str.capitalize()  # 'Male'/'Female'
    df['gt_age'] = df['age']           # as-is, e.g. '0-2'

    # 2. Init model and aligner
    predictor = FairFacePredictor(model_selection=0)
    face_aligner = FaceDetection()
    race_labels = predictor.race_labels_7      # ['White','Black','Latino_Hispanic',...]
    gender_labels = predictor.gender_labels    # ['Male','Female']
    age_labels = predictor.age_labels          # e.g. ['0-2','3-9',...]

    # 3. Predict
    preds = {'race': [], 'gender': [], 'age': []}
    base_path = Path(csv_path).parent
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating 7-class"):
        img_path = base_path / row['file']
        # missing file
        if not img_path.is_file():
            preds['race'].append(None)
            preds['gender'].append(None)
            preds['age'].append(None)
            continue
        # load and align
        img = cv2.imread(str(img_path))
        if img is None:
            preds['race'].append(None)
            preds['gender'].append(None)
            preds['age'].append(None)
            continue
        facechip = face_aligner.detect_and_align(img)
        if facechip is None:
            preds['race'].append(None)
            preds['gender'].append(None)
            preds['age'].append(None)
            continue
        # predict
        res = predictor.predict(facechip)
        r_lbl, _ = res.get('race', (None, None))
        g_lbl, _ = res.get('gender', (None, None))
        a_lbl, _ = res.get('age', (None, None))
        # validate vs label sets
        preds['race'].append(r_lbl if r_lbl in race_labels else None)
        preds['gender'].append(g_lbl if g_lbl in gender_labels else None)
        preds['age'].append(a_lbl if a_lbl in age_labels else None)

    # 4. Attach predictions
    df['pred_race']   = preds['race']
    df['pred_gender'] = preds['gender']
    df['pred_age']    = preds['age']

    # 5. Compute metrics on valid subset
    valid = df.dropna(subset=['pred_race','pred_gender','pred_age']).reset_index(drop=True)
    total = len(valid)

    acc_race   = accuracy_score(valid['gt_race'],   valid['pred_race'])
    acc_gender = accuracy_score(valid['gt_gender'], valid['pred_gender'])
    acc_age    = accuracy_score(valid['gt_age'],    valid['pred_age'])

    print(f"Overall Race accuracy:   {acc_race*100:.2f}% ({(valid['gt_race']==valid['pred_race']).sum()}/{total})")
    print(f"Overall Gender accuracy: {acc_gender*100:.2f}% ({(valid['gt_gender']==valid['pred_gender']).sum()}/{total})")
    print(f"Overall Age accuracy:    {acc_age*100:.2f}% ({(valid['gt_age']==valid['pred_age']).sum()}/{total})")

    if show_classification_report:
        print("\n-- Classification Reports --")
        print("Race:\n", classification_report(valid['gt_race'], valid['pred_race'], labels=race_labels, digits=4))
        print("Gender:\n", classification_report(valid['gt_gender'], valid['pred_gender'], labels=gender_labels, digits=4))
        print("Age:\n", classification_report(valid['gt_age'], valid['pred_age'], labels=age_labels, digits=4))

    print("\n-- Confusion Matrices --")
    cm_r = confusion_matrix(valid['gt_race'], valid['pred_race'], labels=race_labels)
    print(pd.DataFrame(cm_r, index=race_labels, columns=race_labels))

    cm_g = confusion_matrix(valid['gt_gender'], valid['pred_gender'], labels=gender_labels)
    print(pd.DataFrame(cm_g, index=gender_labels, columns=gender_labels))

    cm_a = confusion_matrix(valid['gt_age'], valid['pred_age'], labels=age_labels)
    print(pd.DataFrame(cm_a, index=age_labels, columns=age_labels))

    # 6. Save full results
    out_csv = base_path / out_csv_name
    df.to_csv(out_csv, index=False)
    print(f"\nResults saved to: {out_csv.resolve()}")

def evaluate_model_4(
    model_dir: str = MODEL_PATH,
    csv_path: str = "data/fairface_label_val.csv",
    out_csv_name: str = "fairface_results_4.csv",
    show_classification_report: bool = True,
):
    """
    Evaluate 4-class race-only FairFace model on validation set.
    - In/Out CSV dùng cột 'file' (đường dẫn ảnh), 'race' (nhãn gốc 7 lớp).
    - Chuyển 7 lớp -> 4 lớp: ['White','Black','Asian','Indian'], drop other.
    - In ra accuracy, classification report, confusion matrix.
    - Lưu kết quả kèm cột 'pred_race_4' vào CSV đầu ra.
    """
    import os
    from pathlib import Path
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        classification_report,
    )
    from tqdm import tqdm
    from mdp_aligner import FaceDetection

    # 1. Mapping từ 7->4 cho ground truth
    map7to4 = {
        'White': 'White',
        'Black': 'Black',
        'East Asian': 'Asian',
        'Southeast Asian': 'Asian',
        'Indian': 'Indian',
        # drop classes below by mapping to None
        'Latino_Hispanic': None,
        'Middle Eastern': None,
    }

    # 2. Load ground truth
    df = pd.read_csv(csv_path)
    df['gt_race'] = df['race'].map(map7to4)
    df = df.dropna(subset=['gt_race']).reset_index(drop=True)

    # 3. Init predictor & face aligner
    predictor = FairFacePredictor(model_dir=model_dir, model_selection=1)
    face_aligner = FaceDetection()
    labels4 = predictor.race_labels_4  # ['White','Black','Asian','Indian']

    # 4. Predict
    preds = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        img_file = Path(csv_path).parent / row['file']

        if not img_file.is_file():
            preds.append(None)
            continue

        img = cv2.imread(str(img_file))
        if img is None:
            preds.append(None)
            continue
        facechip = face_aligner.detect_and_align(img)
        if facechip is None:
            preds.append(None)
            continue

        res = predictor.predict(facechip)
        pred_label, _ = res.get('race', (None, None))
        # skip if prediction missing
        if pred_label is None:
            preds.append(None)
            continue

        # Normalize: underscore -> space
        pred_norm = pred_label.replace('_', ' ')
        # Map to 4-class
        if pred_norm in ['East Asian', 'Southeast Asian']:
            pred4 = 'Asian'
        elif pred_norm in labels4:
            pred4 = pred_norm
        else:
            pred4 = None

        preds.append(pred4)

    # 5. Compute metrics
    df['pred_race_4'] = preds
    valid = df.dropna(subset=['pred_race_4']).reset_index(drop=True)

    y_true = valid['gt_race']
    y_pred = valid['pred_race_4']

    acc = accuracy_score(y_true, y_pred)
    print(f"Overall 4-class accuracy: {acc*100:.2f}% "
          f"({(y_true==y_pred).sum()}/{len(valid)})")

    if show_classification_report:
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, digits=4))

    cm = confusion_matrix(y_true, y_pred, labels=labels4)
    cm_df = pd.DataFrame(cm, index=labels4, columns=labels4)
    print("\nConfusion Matrix:")
    print(cm_df)

    # 6. Save full results
    out_csv = Path(csv_path).parent / out_csv_name
    df.to_csv(out_csv, index=False)
    print(f"\nResults saved to: {out_csv.resolve()}")

def show_facechip():
    import pandas as pd    
    df = pd.read_csv("../data/fairface_label_val.csv")
    predictor = FairFacePredictor(model_dir=MODEL_PATH, model_selection=0)
    try:
        for _, row in df.iterrows():
            img_file = os.path.join("../data", row['file'])
            img = cv2.imread(img_file)
            if img is None:
                continue
            # facechip = predictor.detect_facechip(img)     
            cv2.imshow('Detected Faces', img)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        cv2.destroyAllWindows()

def main():
    evaluate_model_4()