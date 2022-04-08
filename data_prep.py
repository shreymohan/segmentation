from skimage import draw
from skimage import io
import os
import json

data_dir=''
masks_dir=''

def poly2mask(blobs, filename, path_to_masks_folder, h, w, label, idx):
    mask = np.zeros((h, w),dtype=np.uint8)
    for l in blobs:
        fill_row_coords, fill_col_coords = draw.polygon(l[1], l[0], l[2])
        mask[fill_row_coords, fill_col_coords] = 1
    io.imsave(path_to_masks_folder + "/" + str(filename) + ".png", mask)


for annotation in sorted(os.listdir(data_dir+'/annotations')):
  f = open(data_dir+'/annotations/'+annotation)

  classes = {}
  data = json.load(f)
      #train.append(data)
  #c = 0
  filename=annotation.split('.')[0]
  for obj in data['shapes']:
    blobs = []
    label = obj['label']
    if label not in classes:
      classes[label] = 0
    points = obj['points']  
    h = data['imageHeight']
    w = data['imageWidth']
    x_coord = []
    y_coord = []
    l=[]
    for p in points:
      x_coord.append(p[0])
      y_coord.append(p[1])
    shape = (h, w)
    l.append(x_coord)
    l.append(y_coord)
    l.append(shape)
    blobs.append(l)
    poly2mask(blobs, filename, masks_dir, data['imageHeight'], data['imageWidth'], label,
              classes[label])  
