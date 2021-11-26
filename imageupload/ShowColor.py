import cv2
import math
import numpy as np
# coding: utf-8
# import the necessary packages
from imutils import face_utils
import dlib
import matplotlib.pyplot as plt
import sys
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import io
from itertools import compress
from scipy.spatial import distance
import copy
import operator

class white_balance:
    def apply_mask(matrix, mask, fill_value):
        masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
        return masked.filled()

    def apply_threshold(matrix, low_value, high_value):
        low_mask = matrix < low_value
        matrix = apply_mask(matrix, low_mask, low_value)

        high_mask = matrix > high_value
        matrix = apply_mask(matrix, high_mask, high_value)

        return matrix

    def balancing(img, percent):
        assert img.shape[2] == 3
        assert percent > 0 and percent < 100

        half_percent = percent / 200.0

        channels = cv2.split(img)

        out_channels = []
        for channel in channels:
            assert len(channel.shape) == 2
            # find the low and high precentile values (based on the input percentile)
            height, width = channel.shape
            vec_size = width * height
            flat = channel.reshape(vec_size)

            assert len(flat.shape) == 1

            flat = np.sort(flat)

            n_cols = flat.shape[0]

            low_val = flat[math.floor(n_cols * half_percent)]
            high_val = flat[math.ceil(n_cols * (1.0 - half_percent))]

            print("Lowval: ", low_val)
            print("Highval: ", high_val)

            # saturate below the low percentile and above the high percentile
            thresholded = apply_threshold(channel, low_val, high_val)
            # scale the channel
            normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
            out_channels.append(normalized)
        out = cv2.merge(out_channels)
        cv2.imwrite("wb.jpg", out)
        return out




class DetectFace:

    def __init__(self, image):
        # initialize dlib's face detector (HOG-based)
        # and then create the facial landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('../res/shape_predictor_68_face_landmarks.dat')

        # face detection part
        self.img = cv2.imread(image)
        # if self.img.shape[0]>500:
        #  self.img = cv2.resize(self.img, dsize=(0,0), fx=0.8, fy=0.8)

        # init face parts
        self.right_eyebrow = []
        self.left_eyebrow = []
        self.right_eye = []
        self.left_eye = []
        self.left_cheek = []
        self.right_cheek = []

        # detect the face parts and set the variables
        self.detect_face_part()

    # return type : np.array
    def detect_face_part(self):
        face_parts = [[], [], [], [], [], [], []]
        face_parts.append(10)
        # detect faces in the grayscale image
        rect = self.detector(cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY), 1)[0]

        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = self.predictor(cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY), rect)
        shape = face_utils.shape_to_np(shape)

        idx = 0
        # loop over the face parts individually
        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            face_parts[idx] = shape[i:j]
            idx += 1
        face_parts = face_parts[1:6]
        # set the variables
        # Caution: this coordinates fits on the RESIZED image.
        self.right_eyebrow = self.extract_face_part(face_parts[2])
        # cv2.imshow("right_eyebrow", self.right_eyebrow)
        # cv2.waitKey(0)
        self.left_eyebrow = self.extract_face_part(face_parts[1])
        # cv2.imshow("right_eyebrow", self.left_eyebrow)
        # cv2.waitKey(0)
        self.right_eye = self.extract_face_part(face_parts[3])
        # cv2.imshow("right_eyebrow", self.right_eye)
        # cv2.waitKey(0)
        self.left_eye = self.extract_face_part(face_parts[4])
        # cv2.imshow("right_eyebrow", self.left_eye)
        # cv2.waitKey(0)
        # Cheeks are detected by relative position to the face landmarks
        self.left_cheek = self.img[shape[29][1]:shape[33][1], shape[4][0]:shape[48][0]]
        # cv2.imshow("right_eyebrow", self.left_cheek)
        # cv2.waitKey(0)
        self.right_cheek = self.img[shape[29][1]:shape[33][1], shape[54][0]:shape[12][0]]
        # cv2.imshow("right_eyebrow", self.right_cheek)
        # cv2.waitKey(0)

    # parameter example : self.right_eye
    # return type : image
    def extract_face_part(self, face_part_points):
        (x, y, w, h) = cv2.boundingRect(face_part_points)
        crop = self.img[y:y + h, x:x + w]
        adj_points = np.array([np.array([p[0] - x, p[1] - y]) for p in face_part_points])

        # Create an mask
        mask = np.zeros((crop.shape[0], crop.shape[1]))
        cv2.fillConvexPoly(mask, adj_points, 1)
        mask = mask.astype(np.bool)
        crop[np.logical_not(mask)] = [255, 0, 0]

        return crop


class DominantColors:

    CLUSTERS = None
    IMAGE = None
    COLORS = None
    LABELS = None

    def __init__(self, image, clusters=3):
        self.CLUSTERS = clusters
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.IMAGE = img.reshape((img.shape[0] * img.shape[1], 3))

        #using k-means to cluster pixels
        kmeans = KMeans(n_clusters = self.CLUSTERS)
        kmeans.fit(self.IMAGE)

        #the cluster centers are our dominant colors.
        self.COLORS = kmeans.cluster_centers_
        self.LABELS = kmeans.labels_

    def rgb_to_hex(self, rgb):
        return '#%02x%02x%02x' % (int(rgb[0]), int(rgb[1]), int(rgb[2]))

    # Return a list in order of color that appeared most often.
    def getHistogram(self):
        numLabels = np.arange(0, self.CLUSTERS+1)
        #create frequency count tables
        (hist, _) = np.histogram(self.LABELS, bins = numLabels)
        hist = hist.astype("float")
        hist /= hist.sum()

        colors = self.COLORS
        #descending order sorting as per frequency count
        colors = colors[(-hist).argsort()]
        hist = hist[(-hist).argsort()]
        for i in range(self.CLUSTERS):
            colors[i] = colors[i].astype(int)
        # Blue mask 제거
        fil = [colors[i][2] < 250 and colors[i][0] > 10 for i in range(self.CLUSTERS)]
        colors = list(compress(colors, fil))
        return colors, hist

    def plotHistogram(self):
        colors, hist = self.getHistogram()
        #creating empty chart
        chart = np.zeros((50, 500, 3), np.uint8)
        start = 0

        #creating color rectangles
        for i in range(len(colors)):
            end = start + hist[i] * 500
            r,g,b = colors[i]
            #using cv2.rectangle to plot colors
            cv2.rectangle(chart, (int(start), 0), (int(end), 50), (r,g,b), -1)
            start = end

        #display chart
        plt.figure()
        plt.axis("off")
        plt.imshow(chart)
        plt.show()

        return colors

class tone_analysis:
    def is_warm(lab_b, a):
        '''
        파라미터 lab_b = [skin_b, hair_b, eye_b]
        a = 가중치 [skin, hair, eye]
        질의색상 lab_b값에서 warm의 lab_b, cool의 lab_b값 간의 거리를
        각각 계산하여 warm이 가까우면 1, 반대 경우 0 리턴
        '''
        # standard of skin, eyebrow, eye
        warm_b_std = [11.6518, 11.71445, 3.6484]
        cool_b_std = [4.64255, 4.86635, 0.18735]

        warm_dist = 0
        cool_dist = 0

        body_part = ['skin', 'eyebrow', 'eye']
        for i in range(3):
            warm_dist += abs(lab_b[i] - warm_b_std[i]) * a[i]
            print(body_part[i], "의 warm 기준값과의 거리")
            print(abs(lab_b[i] - warm_b_std[i]))
            cool_dist += abs(lab_b[i] - cool_b_std[i]) * a[i]
            print(body_part[i], "의 cool 기준값과의 거리")
            print(abs(lab_b[i] - cool_b_std[i]))
        if (warm_dist <= cool_dist):
            return 1  # warm
        else:
            return 0  # cool

    def is_spr(hsv_s, a):
        '''
        파라미터 hsv_s = [skin_s, hair_s, eye_s]
        a = 가중치 [skin, hair, eye]
        질의색상 hsv_s값에서 spring의 hsv_s, fall의 hsv_s값 간의 거리를
        각각 계산하여 spring이 가까우면 1, 반대 경우 0 리턴
        '''
        # skin, hair, eye
        spr_s_std = [18.59296, 30.30303, 25.80645]
        fal_s_std = [27.13987, 39.75155, 37.5]

        spr_dist = 0
        fal_dist = 0

        body_part = ['skin', 'eyebrow', 'eye']
        for i in range(3):
            spr_dist += abs(hsv_s[i] - spr_s_std[i]) * a[i]
            print(body_part[i], "의 spring 기준값과의 거리")
            print(abs(hsv_s[i] - spr_s_std[i]) * a[i])
            fal_dist += abs(hsv_s[i] - fal_s_std[i]) * a[i]
            print(body_part[i], "의 fall 기준값과의 거리")
            print(abs(hsv_s[i] - fal_s_std[i]) * a[i])

        if (spr_dist <= fal_dist):
            return 1  # spring
        else:
            return 0  # fall

    def is_smr(hsv_s, a):
        '''
        파라미터 hsv_s = [skin_s, hair_s, eye_s]
        a = 가중치 [skin, hair, eye]
        질의색상 hsv_s값에서 summer의 hsv_s, winter의 hsv_s값 간의 거리를
        각각 계산하여 summer가 가까우면 1, 반대 경우 0 리턴
        '''
        # skin, eyebrow, eye
        smr_s_std = [12.5, 21.7195, 24.77064]
        wnt_s_std = [16.73913, 24.8276, 31.3726]
        a[1] = 0.5  # eyebrow 영향력 적기 때문에 가중치 줄임

        smr_dist = 0
        wnt_dist = 0

        body_part = ['skin', 'eyebrow', 'eye']
        for i in range(3):
            smr_dist += abs(hsv_s[i] - smr_s_std[i]) * a[i]
            print(body_part[i], "의 summer 기준값과의 거리")
            print(abs(hsv_s[i] - smr_s_std[i]) * a[i])
            wnt_dist += abs(hsv_s[i] - wnt_s_std[i]) * a[i]
            print(body_part[i], "의 winter 기준값과의 거리")
            print(abs(hsv_s[i] - wnt_s_std[i]) * a[i])

        if (smr_dist <= wnt_dist):
            return 1  # summer
        else:
            return 0  # winter

