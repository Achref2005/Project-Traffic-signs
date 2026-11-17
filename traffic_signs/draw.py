import cv2


def draw_dets(img, dets):
    """Draw boxes with class, confidence and (video mode) track id / LSTM-refined class."""
    out = img.copy()
    for d in dets:
        x1, y1, x2, y2 = map(int, d["box"])
        name = d.get("refined_class", d["yolo_class"])
        conf = d.get("refined_conf", d["yolo_conf"])
        label = (f"#{d['track_id']} " if "track_id" in d else "") + f"{name} {conf:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(out, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
    return out
