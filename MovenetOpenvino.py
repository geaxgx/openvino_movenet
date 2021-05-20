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

# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# LINES_*_BODY are used when drawing the skeleton onto the source image. 
# Each variable is a list of continuous lines.
# Each line is a list of keypoints as defined at https://github.com/tensorflow/tfjs-models/tree/master/pose-detection#keypoint-diagram

LINES_BODY = [[4,2],[2,0],[0,1],[1,3],
                [10,8],[8,6],[6,5],[5,7],[7,9],
                [6,12],[12,11],[11,5],
                [12,14],[14,16],[11,13],[13,15]]

class Body:
    def __init__(self, scores=None, keypoints_norm=None):
        self.scores = scores # scores of the keypoints
        self.keypoints_norm = keypoints_norm # Keypoints normalized ([0,1]) coordinates (x,y) in the squared input image
        self.keypoints = None # keypoints coordinates (x,y) in pixels in the source image

    def print(self):
        attrs = vars(self)
        print('\n'.join("%s: %s" % item for item in attrs.items()))

CropRegion = namedtuple('CropRegion',['xmin', 'ymin', 'xmax',  'ymax', 'size']) # All values are in pixel. The region is a square of size 'size' pixels



    

class MovenetOpenvino:
    def __init__(self, input_src=None,
                xml=DEFAULT_MODEL, 
                device="CPU",
                score_thresh=0.2,
                output=None):
        
        self.score_thresh = score_thresh
         
        if input_src.endswith('.jpg') or input_src.endswith('.png') :
            self.input_type= "image"
            self.img = cv2.imread(input_src)
            self.video_fps = 25
            self.img_h, self.img_w = self.img.shape[:2]
        else:
            self.input_type = "video"
            if input_src.isdigit():
                input_type = "webcam"
                input_src = int(input_src)
            self.cap = cv2.VideoCapture(input_src)
            self.video_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            self.img_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.img_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("Video FPS:", self.video_fps)
    
        # Load Openvino models
        self.load_model(xml, device)     

        # Rendering flags
        self.show_fps = True
        self.show_crop = False

        if output is None: 
            self.output = None
        else:
            if self.input_type == "image":
                # For an source image, we will output one image (and not a video) and exit
                self.output = output
            else:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                self.output = cv2.VideoWriter(output,fourcc,self.video_fps,(self.img_w, self.img_h)) 

        # Defines the default crop region (pads the full image from both sides to make it a square image) 
        # Used when the algorithm cannot reliably determine the crop region from the previous frame.
        box_size = max(self.img_w, self.img_h)
        x_min = (self.img_w - box_size) // 2
        y_min = (self.img_h - box_size) // 2
        self.init_crop_region = CropRegion(x_min, y_min, x_min+box_size, y_min+box_size, box_size)
        print("init crop", self.init_crop_region)
        
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

    def crop_and_resize(self, frame, crop_region):
        """Crops and resize the image to prepare for the model input."""
        cropped = frame[max(0,crop_region.ymin):min(self.img_h,crop_region.ymax), max(0,crop_region.xmin):min(self.img_w,crop_region.xmax)]
        
        if crop_region.xmin < 0 or crop_region.xmax >= self.img_w or crop_region.ymin < 0 or crop_region.ymax >= self.img_h:
            # Padding is necessary        
            cropped = cv2.copyMakeBorder(cropped, 
                                        max(0,-crop_region.ymin), 
                                        max(0, crop_region.ymax-self.img_h),
                                        max(0,-crop_region.xmin), 
                                        max(0, crop_region.xmax-self.img_w),
                                        cv2.BORDER_CONSTANT)

        cropped = cv2.resize(cropped, (self.pd_w, self.pd_h), interpolation=cv2.INTER_AREA)
        return cropped

    def torso_visible(self, scores):
        """Checks whether there are enough torso keypoints.

        This function checks whether the model is confident at predicting one of the
        shoulders/hips which is required to determine a good crop region.
        """
        return ((scores[KEYPOINT_DICT['left_hip']] > self.score_thresh or
                scores[KEYPOINT_DICT['right_hip']] > self.score_thresh) and
                (scores[KEYPOINT_DICT['left_shoulder']] > self.score_thresh or
                scores[KEYPOINT_DICT['right_shoulder']] > self.score_thresh))

    def determine_torso_and_body_range(self, keypoints, scores, center_x, center_y):
        """Calculates the maximum distance from each keypoints to the center location.

        The function returns the maximum distances from the two sets of keypoints:
        full 17 keypoints and 4 torso keypoints. The returned information will be
        used to determine the crop size. See determine_crop_region for more detail.
        """
        # import pdb
        # pdb.set_trace()
        torso_joints = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        max_torso_yrange = 0.0
        max_torso_xrange = 0.0
        for joint in torso_joints:
            dist_y = abs(center_y - keypoints[KEYPOINT_DICT[joint]][1])
            dist_x = abs(center_x - keypoints[KEYPOINT_DICT[joint]][0])
            if dist_y > max_torso_yrange:
                max_torso_yrange = dist_y
            if dist_x > max_torso_xrange:
                max_torso_xrange = dist_x

        max_body_yrange = 0.0
        max_body_xrange = 0.0
        for i in range(len(KEYPOINT_DICT)):
            if scores[i] < self.score_thresh:
                continue
            dist_y = abs(center_y - keypoints[i][1])
            dist_x = abs(center_x - keypoints[i][0])
            if dist_y > max_body_yrange:
                max_body_yrange = dist_y
            if dist_x > max_body_xrange:
                max_body_xrange = dist_x

        return [max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange]

    def determine_crop_region(self, body):
        """Determines the region to crop the image for the model to run inference on.

        The algorithm uses the detected joints from the previous frame to estimate
        the square region that encloses the full body of the target person and
        centers at the midpoint of two hip joints. The crop size is determined by
        the distances between each joints and the center point.
        When the model is not confident with the four torso joint predictions, the
        function returns a default crop which is the full image padded to square.
        """
        if self.torso_visible(body.scores):
            center_x = (body.keypoints[KEYPOINT_DICT['left_hip']][0] + body.keypoints[KEYPOINT_DICT['right_hip']][0]) // 2
            center_y = (body.keypoints[KEYPOINT_DICT['left_hip']][1] + body.keypoints[KEYPOINT_DICT['right_hip']][1]) // 2
            max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange = self.determine_torso_and_body_range(body.keypoints, body.scores, center_x, center_y)
            crop_length_half = np.amax([max_torso_xrange * 1.9, max_torso_yrange * 1.9, max_body_yrange * 1.2, max_body_xrange * 1.2])
            tmp = np.array([center_x, self.img_w - center_x, center_y, self.img_h - center_y])
            crop_length_half = int(round(np.amin([crop_length_half, np.amax(tmp)])))
            crop_corner = [center_x - crop_length_half, center_y - crop_length_half]

            if crop_length_half > max(self.img_w, self.img_h) / 2:
                return self.init_crop_region
            else:
                crop_length = crop_length_half * 2
                return CropRegion(crop_corner[0], crop_corner[1], crop_corner[0]+crop_length, crop_corner[1]+crop_length,crop_length)
        else:
            return self.init_crop_region

    def pd_postprocess(self, inference, crop_region):
        kps = np.squeeze(inference[self.pd_kps]) # 17x3
        # kps = np.where(kps<0, kps+1, kps) # Bug with Openvino 2021.2
        body = Body(scores=kps[:,2], keypoints_norm=kps[:,[1,0]])
        body.keypoints = (np.array([crop_region.xmin, crop_region.ymin]) + body.keypoints_norm * crop_region.size).astype(np.int)
        body.next_crop_region = self.determine_crop_region(body)
        return body
        

    def pd_render(self, frame, body, crop_region):

        lines = [np.array([body.keypoints[point] for point in line]) for line in LINES_BODY if body.scores[line[0]] > self.score_thresh and body.scores[line[1]] > self.score_thresh]
        cv2.polylines(frame, lines, False, (255, 180, 90), 2, cv2.LINE_AA)
        
        for i,x_y in enumerate(body.keypoints):
            if body.scores[i] > self.score_thresh:
                if i % 2 == 1:
                    color = (0,255,0) 
                elif i == 0:
                    color = (0,255,255)
                else:
                    color = (0,0,255)
                cv2.circle(frame, (x_y[0], x_y[1]), 4, color, -11)

        if self.show_crop:
            cv2.rectangle(frame, (crop_region.xmin, crop_region.ymin), (crop_region.xmax, crop_region.ymax), (0,255,255), 2)

                
    def run(self):

        self.fps = FPS()

        nb_pd_inferences = 0
        glob_pd_rtrip_time = 0

        use_previous_keypoints = False

        crop_region = self.init_crop_region

        while True:
                
            if self.input_type == "image":
                frame = self.img.copy()
            else:
                ok, frame = self.cap.read()
                if not ok:
                    break

            cropped = self.crop_and_resize(frame, crop_region)
                
            frame_nn = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB).astype(np.float32)[None,] 
            pd_rtrip_time = now()
            inference = self.pd_exec_net.infer(inputs={self.pd_input_blob: frame_nn})
            glob_pd_rtrip_time += now() - pd_rtrip_time
            body = self.pd_postprocess(inference, crop_region)
            self.pd_render(frame, body, crop_region)
            crop_region = body.next_crop_region
            nb_pd_inferences += 1

            self.fps.update()               

            if self.show_fps:
                self.fps.draw(frame, orig=(50,50), size=1, color=(240,180,100))
            cv2.imshow("Movepose", frame)

            if self.output:
                if self.input_type == "image":
                    cv2.imwrite(self.output, frame)
                    break
                else:
                    self.output.write(frame)

            key = cv2.waitKey(1) 
            if key == ord('q') or key == 27:
                break
            elif key == 32:
                # Pause on space bar
                cv2.waitKey(0)
            elif key == ord('f'):
                self.show_fps = not self.show_fps
            elif key == ord('c'):
                self.show_crop = not self.show_crop

        # Print some stats
        if nb_pd_inferences > 1:
            global_fps, nb_frames = self.fps.get_global()

            print(f"FPS : {global_fps:.1f} f/s (# frames = {nb_frames})")
            print(f"# pose detection inferences : {nb_pd_inferences}")
            print(f"Pose detection round trip   : {glob_pd_rtrip_time/nb_pd_inferences*1000:.1f} ms")

        if self.output and self.input_type != "image":
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
    parser.add_argument("-s", "--score_threshold", default=0.2, type=float,
                        help="Confidence score to determine whether a keypoint prediction is reliable (default=%(default)f)")                     
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
                    output=args.output)
    pd.run()
