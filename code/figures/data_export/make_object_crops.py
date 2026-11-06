"""Object gallery crops for the dataset figure: for every object, the most visible test instance
(visib_fract >= 0.9 preferred, else >= 0.5; largest visible pixel count), cropped around bbox_visib; pixels are not modified apart from resizing.
Usage: python make_object_crops.py <bop_test_dir> <out_dir> <size>"""
import json, sys, os
from pathlib import Path
import numpy as np, cv2
root, out, S = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
best = {}
for sc in sorted(p for p in root.iterdir() if p.is_dir()):
    gt = json.load(open(sc/'scene_gt.json')); gi = json.load(open(sc/'scene_gt_info.json'))
    for im, insts in gt.items():
        for k, g in enumerate(insts):
            inf = gi[im][k]
            if inf['visib_fract'] < 0.5: continue
            o = g['obj_id']; key = (inf['visib_fract'] >= 0.9, inf['px_count_visib'])
            if o not in best or key > best[o][0]:
                best[o] = (key, sc, int(im), k, inf['bbox_visib'])
rows = []
for o, (px, sc, im, k, (x, y, w, h)) in sorted(best.items()):
    rgbp = next((sc/'rgb'/f'{im:06d}{e}') for e in ('.png', '.jpg') if (sc/'rgb'/f'{im:06d}{e}').exists())
    img = cv2.imread(str(rgbp)); m = np.ones(img.shape[:2], bool)  # mask no longer used (no background edit)
    s = int(max(w, h) * 1.18); cx, cy = x + w / 2, y + h / 2
    x0, y0 = int(round(cx - s / 2)), int(round(cy - s / 2))
    pad = s
    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    m = cv2.copyMakeBorder(m.astype(np.uint8), pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0) > 0
    c = img[y0 + pad:y0 + pad + s, x0 + pad:x0 + pad + s].astype(float); mm = m[y0 + pad:y0 + pad + s, x0 + pad:x0 + pad + s]
    c = cv2.resize(c.astype(np.uint8), (S, S), interpolation=cv2.INTER_AREA if s > S else cv2.INTER_CUBIC)
    cv2.imwrite(str(out/f'obj_{o:02d}.png'), c)
    rows.append(f'{o},{sc.name},{im},{k},{px[1]},{int(px[0])}')
open(out/'crops_source.csv', 'w').write('object_id,scene_id,img_id,inst,px_visib,visib_ge_0.9\n' + '\n'.join(rows) + '\n')
print(len(rows))
