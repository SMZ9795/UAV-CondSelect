"""Explicit XML/YOLO to COCO box conversion; no implicit origin adjustment."""
import argparse,json,xml.etree.ElementTree as ET
from pathlib import Path

def xml_boxes(path,target_width,target_height):
 root=ET.parse(path).getroot();W=float(root.findtext('size/width'));H=float(root.findtext('size/height'));result=[]
 for obj in root.findall('object'):
  box=obj.find('bndbox');x1,y1,x2,y2=[float(box.findtext(k)) for k in ['xmin','ymin','xmax','ymax']]
  result.append([x1*target_width/W,y1*target_height/H,(x2-x1)*target_width/W,(y2-y1)*target_height/H])
 return result

def yolo_boxes(path,W,H):
 result=[]
 for line in Path(path).read_text().splitlines():
  cls,x,y,w,h=map(float,line.split());assert cls==0;result.append([(x-w/2)*W,(y-h/2)*H,w*W,h*H])
 return result

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--format',choices=['xml','yolo'],required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--width',type=int,required=True);p.add_argument('--height',type=int,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--image-file',required=True);p.add_argument('--image-id',type=int,default=1);p.add_argument('--source-id');a=p.parse_args();boxes=(xml_boxes if a.format=='xml' else yolo_boxes)(a.input,a.width,a.height)
 image={'id':a.image_id,'file_name':a.image_file,'width':a.width,'height':a.height}
 if a.source_id:image['base_source_identity']=a.source_id
 annotations=[{'id':i+1,'image_id':a.image_id,'category_id':1,'bbox':b,'area':b[2]*b[3],'iscrowd':0} for i,b in enumerate(boxes)]
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'images':[image],'annotations':annotations,'categories':[{'id':1,'name':'UAV','supercategory':'aircraft'}]},indent=2)+'\n')
if __name__=='__main__':main()
