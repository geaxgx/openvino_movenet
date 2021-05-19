import numpy as np
from collections import namedtuple
import cv2
from pathlib import Path
from FPS import FPS, now
import argparse
import os
from openvino.inference_engine import IENetwork, IECore


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = SCRIPT_DIR / "models/movenet_singlepose_lightning_FP32.xml"


# LINES_*_BODY are used when drawing the skeleton onto the source image. 
# Each variable is a list of continuous lines.
# Each line is a list of keypoints as defined at https://github.com/tensorflow/tfjs-models/tree/master/pose-detection#keypoint-diagram

LINES_BODY = [[4,2],[2,0],[0,1],[1,3],
                [10,8],[8,6],[6,5],[5,7],[7,9],
                [6,12],[12,11],[11,5],
                [12,14],[14,16],[11,13],[13,15]]

class Region:
    # One region per body detected. With this version of Movenet, one and only one body 
    def __init__(self, scores=None, landmarks_norm=None):
        self.scores = scores # scores of the landmarks
        self.landmarks_norm = landmarks_norm # Landmark normalized ([0,1]) coordinates (x,y) in the squared input image
        self.landmarks = None # Landmarks coordinates (x,y) in pixels in the source image

    def print(self):
        attrs = vars(self)
        print('\n'.join("%s: %s" % item for item in attrs.items()))


class MovenetOpenvino:
    def __init__(self, input_src=None,
                xml=DEFAULT_MODEL, 
                device="CPU",
                score_thresh=0.15,
                crop=False,
                output=None):
        
        self.score_thresh = score_thresh
        self.crop = crop
         
        if input_src.endswith('.jpg') or input_src.endswith('.png') :
            self.input_type= "image"
            self.img = cv2.imread(input_src)
            self.video_fps = 25
            video_height, video_width = self.img.shape[:2]
        else:
            self.input_type = "video"
            if input_src.isdigit():
                input_type = "webcam"
                input_src = int(input_src)
            self.cap = cv2.VideoCapture(input_src)
            self.video_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("Video FPS:", self.video_fps)

        # 17 landmarks
        self.nb_lms = 17
    
        # Load Openvino models
        self.load_model(xml, device)     

        # Rendering flags
        self.show_fps = True



        if output is None:
            self.output = None
        else:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self.output = cv2.VideoWriter(output,fourcc,self.video_fps,(video_width, video_height)) 

    def load_model(self, xml_path, device):

        print("Loading Inference Engine")
        self.ie = IECore()
        print("Device info:")
        versions = self.ie.get_versions(device)
        print("{}{}".format(" "*8, device))
        print("{}MKLDNNPlugin version ......... {}.{}".format(" "*8, versions[device].major, versions[device].minor))
        print("{}Build ........... {}".format(" "*8, versions[device].build_number))

        name = os.path.splitext(xml_path)[0]
        bin_path = name + '.bin'
        print("Pose Detection model - Reading network files:\n\t{}\n\t{}".format(xml_path, bin_path))
        self.pd_net = self.ie.read_network(model=xml_path, weights=bin_path)
        # Input blob: input:0 - shape: [1, 192, 192, 3] (for lightning)
        # Input blob: input:0 - shape: [1, 256, 256, 3] (for thunder)
        # Output blob: 7022.0 - shape: [1, 1, 1]
        # Output blob: 7026.0 - shape: [1, 1, 17]
        # Output blob: Identity - shape: [1, 1, 17, 3]
        self.pd_input_blob = next(iter(self.pd_net.input_info))
        print(f"Input blob: {self.pd_input_blob} - shape: {self.pd_net.input_info[self.pd_input_blob].input_data.shape}")
        _,self.pd_h,self.pd_w,_ = self.pd_net.input_info[self.pd_input_blob].input_data.shape
        for o in self.pd_net.outputs.keys():
            print(f"Output blob: {o} - shape: {self.pd_net.outputs[o].shape}")
        self.pd_kps = "Identity"
        print("Loading pose detection model into the plugin")
        self.pd_exec_net = self.ie.load_network(network=self.pd_net, num_requests=1, device_name=device)

        self.infer_nb = 0
        self.infer_time_cumul = 0

         
    def pd_postprocess(self, inference):
        kps = np.squeeze(inference[self.pd_kps]) # 17x3
        region = Region(scores=kps[:,2], landmarks_norm=kps[:,[1,0]])
        region.landmarks = (region.landmarks_norm * self.frame_size).astype(np.int)
        if self.pad_h > 0:
            region.landmarks[:,1] -= self.pad_h
        if self.pad_w > 0:
            region.landmarks[:,0] -= self.pad_w
        self.regions = [region]
        

    def pd_render(self, frame, region):

        lines = [np.array([region.landmarks[point] for point in line]) for line in LINES_BODY if region.scores[line[0]] > self.score_thresh and region.scores[line[1]] > self.score_thresh]
        cv2.polylines(frame, lines, False, (255, 180, 90), 2, cv2.LINE_AA)
        
        for i,x_y in enumerate(region.landmarks):
            if region.scores[i] > self.score_thresh:
                if i % 2 == 1:
                    color = (0,255,0) 
                elif i == 0:
                    color = (0,255,255)
                else:
                    color = (0,0,255)
                cv2.circle(frame, (x_y[0], x_y[1]), 4, color, -11)

                
    def run(self):

        self.fps = FPS()

        nb_pd_inferences = 0
        glob_pd_rtrip_time = 0

        get_new_frame = True
        use_previous_landmarks = False

        while True:
            if get_new_frame:
                
                if self.input_type == "image":
                    src_frame = self.img.copy()
                else:
                    ok, src_frame = self.cap.read()
                    if not ok:
                        break
                h, w = src_frame.shape[:2]
                if self.crop:
                    # Cropping the long side to get a square shape
                    self.frame_size = min(h, w)
                    dx = (w - self.frame_size) // 2
                    dy = (h - self.frame_size) // 2
                    self.pad_h = self.pad_w = 0
                    square_frame = src_frame = src_frame[dy:dy+self.frame_size, dx:dx+self.frame_size]
                else:
                    # Padding on the small side to get a square shape
                    self.frame_size = max(h, w)
                    self.pad_h = int((self.frame_size - h)/2)
                    self.pad_w = int((self.frame_size - w)/2)
                    square_frame = cv2.copyMakeBorder(src_frame, self.pad_h, self.pad_h, self.pad_w, self.pad_w, cv2.BORDER_CONSTANT)

            if False: # use_previous_landmarks:
                self.regions = regions_from_landmarks
                mpu.detections_to_rect(self.regions, kp_pair=[0,1]) # self.regions.pd_kps are initialized from landmarks on previous frame
                mpu.rect_transformation(self.regions, self.frame_size, self.frame_size)
            else:
                # Infer pose detection
                # Resize image to NN square input shape
                frame_nn = cv2.resize(square_frame, (self.pd_w, self.pd_h), interpolation=cv2.INTER_AREA)
                frame_nn = cv2.cvtColor(frame_nn, cv2.COLOR_BGR2RGB).astype(np.float32)[None,] 
    
                pd_rtrip_time = now()
                inference = self.pd_exec_net.infer(inputs={self.pd_input_blob: frame_nn})
                glob_pd_rtrip_time += now() - pd_rtrip_time
                self.pd_postprocess(inference)
                for r in self.regions:
                    self.pd_render(src_frame, r)
                nb_pd_inferences += 1

            self.fps.update()               

            if self.show_fps:
                self.fps.draw(src_frame, orig=(50,50), size=1, color=(240,180,100))
            cv2.imshow("Movepose", src_frame)

            if self.output:
                self.output.write(src_frame)

            key = cv2.waitKey(1) 
            if key == ord('q') or key == 27:
                break
            elif key == 32:
                # Pause on space bar
                cv2.waitKey(0)
            elif key == ord('f'):
                self.show_fps = not self.show_fps

        # Print some stats
        global_fps, nb_frames = self.fps.get_global()
        print(f"FPS : {global_fps:.1f} f/s (# frames = {nb_frames})")
        print(f"# pose detection inferences : {nb_pd_inferences}")
        print(f"Pose detection round trip   : {glob_pd_rtrip_time/nb_pd_inferences*1000:.1f} ms")

        if self.output:
            self.output.release()
           

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='0', 
                        help="Path to video or image file to use as input (default=%(default)s)")
    parser.add_argument("-p", "--precision", type=int, choices=[16, 32], default=32,
                        help="Precision (default=%(default)i")    
    parser.add_argument("-m", "--model", type=str, choices=['lightning', 'thunder'], default='thunder',
                        help="Model to use (default=%(default)s")                  
    parser.add_argument("--xml", type=str,
                        help="Path to an .xml file for model")
    parser.add_argument("-d", "--device", default='CPU', type=str,
                        help="Target device to run the model (default=%(default)s)")  
    parser.add_argument("-t", "--score_threshold", default=0.15, type=float,
                        help="Minimum score threshold for landmarks (default=%(default)i)")  
    parser.add_argument('-c', '--crop', action="store_true", 
                        help="Center crop frames to a square shape before feeding pose detection model")                     
    parser.add_argument("-o","--output",
                        help="Path to output video file")
    
    args = parser.parse_args()

    
    if args.device == "MYRIAD":
        args.precision = 16
    if not args.xml:
        args.xml = SCRIPT_DIR / f"models/movenet_singlepose_{args.model}_FP{args.precision}.xml"

    pd = MovenetOpenvino(input_src=args.input, 
                    xml=args.xml,
                    device=args.device, 
                    score_thresh=args.score_threshold,
                    crop=args.crop,
                    output=args.output)
    pd.run()
