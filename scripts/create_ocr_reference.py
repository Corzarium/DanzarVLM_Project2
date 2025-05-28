#!/usr/bin/env python3
import cv2, json, os
from copy import deepcopy

# globals for mouse callback
rois = {}
drawing = False
ix = iy = 0

def mouse_handler(event, x, y, flags, param):
    global ix, iy, drawing, rois
    img = param["img"]
    disp = deepcopy(img)

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.rectangle(disp, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.imshow("Annotate OCR ROIs", disp)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x0, y0 = min(ix, x), min(iy, y)
        w, h = abs(x - ix), abs(y - iy)
        name = input(f"Name for ROI at ({x0},{y0},{w}×{h}): ").strip() or f"roi_{len(rois)+1}"
        rois[name] = {"x": x0, "y": y0, "width": w, "height": h}
        cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (0, 255, 0), 2)
        cv2.putText(img, name, (x0, y0 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("Annotate OCR ROIs", img)

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Click‐and‐drag to define OCR ROIs.")
    p.add_argument("image", help="Path to your screenshot")
    p.add_argument(
        "--output-image",
        default="ocr_reference.png",
        help="Where to save the annotated reference image"
    )
    p.add_argument(
        "--output-json",
        default="ocr_layout.json",
        help="Where to save the ROI JSON"
    )
    args = p.parse_args()

    # load image
    img = cv2.imread(args.image)
    if img is None:
        print(f"❌ Could not load {args.image}")
        exit(1)

    # set up window & callback
    cv2.namedWindow("Annotate OCR ROIs", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Annotate OCR ROIs", mouse_handler, {"img": img})

    print("""
INSTRUCTIONS:
 • Click and drag to draw a box around the UI area you want OCR’d.
 • When you release, you’ll be prompted to name that region.
 • Repeat for as many regions as you like.
 • Press ‘s’ to save your reference image + JSON, or ‘q’ to quit without saving.
""")

    cv2.imshow("Annotate OCR ROIs", img)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('s'):
            # ensure output dirs exist
            os.makedirs(os.path.dirname(args.output_image) or ".", exist_ok=True)
            with open(args.output_json, "w") as f:
                json.dump(rois, f, indent=2)
            cv2.imwrite(args.output_image, img)
            print(f"✅ Saved annotated image → {args.output_image}")
            print(f"✅ Saved ROIs JSON       → {args.output_json}")
            break
        elif key == ord('q'):
            print("Aborting—no files were written.")
            break

    cv2.destroyAllWindows()
