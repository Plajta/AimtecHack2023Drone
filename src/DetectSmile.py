import cv2
import mediapipe as mp
import numpy as np
import math
import os
import time

#neural network things (Not used)
#import process_dataset
#import Model

#Mediapipe setup
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

#via canonical_face_model_uv_visualization.png
face_landmarks_lower = [78, 95, 88, 178, 87, 14, 317, 405, 318, 324, 308]
face_landmarks_upper = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]

#Pytorch setup
#SMILENet = torch.load(os.getcwd() + "/Best_models/SMILENet0.pth") #best model yet

def DrawSmile(arr_y, arr_x, col, blank):
    for i in range(arr_x.shape[0]):
        cv2.circle(blank, (arr_x[i], arr_y[i]), radius=0, color=col, thickness=3)

def JawComputations(jaw_arr):

    jaw_arr_y = jaw_arr[:, 0]
    jaw_arr_x = jaw_arr[:, 1]

    y_min = np.amin(jaw_arr_y)
    x_min = np.amin(jaw_arr_x)

    y_max = np.amax(jaw_arr_y)
    x_max = np.amax(jaw_arr_x)

    #normalize data
    jaw_arr_y = jaw_arr_y - (y_min - 1)
    jaw_arr_x = jaw_arr_x - (x_min - 1)

    #fit polynomial model
    poly_model = np.poly1d(np.polyfit(jaw_arr_x, jaw_arr_y, 3))
    
    #apply polynomial model on y_min ... to ... y_max space
    x_len = x_max - x_min
    
    space = np.arange(x_len)
    y_space = np.floor(poly_model(space))

    out_x = space + x_min
    out_y = y_space + y_min

    return out_y.astype("uint32"), out_x.astype("uint32")


#variables:
roi_size = 160
val_buffer1, val_buffer2 = [], []
val_buffer_out1 = 0
val_buffer_out2 = 0


drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)

face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)


def DetectSMILE(image):
    image = cv2.flip(image, 1)

    h, w, c = image.shape
    blank = np.zeros((h, w, c))
    ROI = np.zeros((roi_size, roi_size, c))
    OUT = np.zeros((8, 8, 3))

    #create cool looking smiley face
    OUT[2:6, 0, :] = (255, 0, 0)
    OUT[2:6, 7, :] = (255, 0, 0)
    OUT[0, 2:6, :] = (255, 0, 0)
    OUT[7, 2:6, :] = (255, 0, 0)
    OUT[6, 6, :] = (255, 0, 0)
    OUT[1, 1, :] = (255, 0, 0)
    OUT[1, 6, :] = (255, 0, 0)
    OUT[6, 1, :] = (255, 0, 0)
    OUT[2, 5, :] = (0, 255, 0)
    OUT[2, 2, :] = (0, 255, 0)

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            #
            # FACE DRAWING
            #
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_iris_connections_style())
            
            #
            # FACE DETECTION
            #
            X_bot = math.floor(face_landmarks.landmark[234].x * w)
            X_top = math.floor(face_landmarks.landmark[454].x * w)

            Y_bot = math.floor(face_landmarks.landmark[152].y * h)
            Y_top = math.floor(face_landmarks.landmark[10].y * h)

            print(X_bot, X_top, Y_bot, Y_top)

            if ((Y_top > 0) and (Y_bot < image.shape[0])) and ((X_bot > 0) and (X_top < image.shape[1])):

                #just for visualisation
                cv2.rectangle(image, (X_bot, Y_bot), (X_top, Y_top), (255, 0, 0), 2)

                face_center = [(((X_top - X_bot) / 2) + X_bot) * w, (((Y_top - Y_bot) / 2) + Y_bot) * h]

                #
                # SMILE LOCALIZATION
                #
                jaw_low = np.zeros((len(face_landmarks_lower), 2))
                jaw_up = np.zeros((len(face_landmarks_upper), 2))
                
                #lower line
                for i, i_low in enumerate(face_landmarks_lower):
                    X = face_landmarks.landmark[i_low].x
                    Y = face_landmarks.landmark[i_low].y

                    X = math.floor(X * w)
                    Y = math.floor(Y * h)

                    jaw_low[i, 0] = Y
                    jaw_low[i, 1] = X

                #upper line
                for i, i_up in enumerate(face_landmarks_upper):
                    X = face_landmarks.landmark[i_up].x
                    Y = face_landmarks.landmark[i_up].y

                    X = math.floor(X * w)
                    Y = math.floor(Y * h)

                    jaw_up[i, 0] = Y
                    jaw_up[i, 1] = X

                out_y_low, out_x_low = JawComputations(jaw_low)
                out_y_up, out_x_up = JawComputations(jaw_up)
            
                #
                # DRAW THAT SMILE
                #
                DrawSmile(out_y_low, out_x_low, (0, 0, 255), blank)
                DrawSmile(out_y_up, out_x_up, (0, 0, 255), blank)

                #
                # DEGRADE SMILE TO 8x8 MATRIX (actual size should be 6x3)
                #
                
                #borders of ROI
                top_x = np.amin(out_x_up) - 5
                top_y = np.amin(out_y_up) - 5
                bot_x = np.amax(out_x_low) + 5
                bot_y = np.amax(out_y_low) + 5

                ROI = blank[top_y:bot_y, top_x:bot_x]

                ROI_mid_R = ROI[:, round(ROI.shape[1] / 2), 2]
                num_R = np.sum(ROI_mid_R == 255)
                num_B = ROI.shape[0] - num_R
                ratio = round(num_R / num_B, 2)

                #get two side points
                pt1 = np.where(ROI[:, 6, 2] == 255)
                pt2 = np.where(ROI[:, ROI.shape[1] - 6, 2] == 255)
                pt1 = round(np.sum(pt1[0]) / pt1[0].shape[0])
                pt2 = round(np.sum(pt2[0]) / pt2[0].shape[0])

                pt1_ratio = round(pt1 / ROI.shape[0], 2)
                pt2_ratio = round(pt2 / ROI.shape[0], 2)
                pt_ratio = round((pt1_ratio + pt2_ratio) / 2, 2)

                #Add to value buffer
                if len(val_buffer1) == 3:
                    val_buffer_out1 = sum(val_buffer1) / len(val_buffer1)
                    val_buffer_out2 = sum(val_buffer2) / len(val_buffer2)
                    val_buffer1 = []
                    val_buffer2 = []
                else:
                    val_buffer1.append(ratio)
                    val_buffer2.append(pt_ratio)


                if 0.6 < val_buffer_out1 < 1:
                    OUT[5, 2:6, :] = (0, 0, 255)
                    print("Mood: 1")
                if 0.3 < val_buffer_out1 < 0.6:
                    if val_buffer_out2 <= 0.3:
                        OUT[6, 2:6, :] = (0, 0, 255)
                        OUT[5, 1, :] = (0, 0, 255)
                        OUT[5, 6, :] = (0, 0, 255)
                        print("Mood: 2, 1")

                    elif val_buffer_out2 <= 0.5:
                        OUT[6, 3:5, :] = (0, 0, 255)
                        OUT[5, 2, :] = (0, 0, 255)
                        OUT[5, 5, :] = (0, 0, 255)
                        print("Mood: 2, 2")
                    else:
                        OUT[5, 3:5, :] = (0, 0, 255)
                        OUT[6, 2, :] = (0, 0, 255)
                        OUT[6, 5, :] = (0, 0, 255)
                        print("Mood: 2, 3")
                if 0.0 < val_buffer_out1 < 0.3:
                    OUT[6, 2:6, :] = (0, 0, 255)
                    OUT[5, 1, :] = (0, 0, 255)
                    OUT[5, 6, :] = (0, 0, 255)
                    OUT[4, 2:6, :] = (0, 0, 255)
                    print("Mood: 3")

                """ Fuck neural nets
                roi_h = bot_y-top_y
                roi_w = bot_x-top_x
                
                #transform ROI into 160x160 full ROI

                scale_y_to_x = roi_h / roi_w
                if scale_y_to_x > 1:
                    scale_y_to_x = 1

                ROI_propose = cv2.resize(blank[top_y:bot_y, top_x:bot_x], (roi_size, int(roi_size * scale_y_to_x)))

                pad_y = int((roi_size - ROI_propose.shape[0]) / 2)

                ROI[pad_y:ROI_propose.shape[0]+pad_y, :] = ROI_propose
                ROI = ROI.astype(dtype=np.uint8)

                #ROI[np.all(ROI == (0, 0, 255), axis=-1)] = (255,255,255)
                #NN_input = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)

                ROI_tensor = process_dataset.Convert_To_Tensor(NN_input)
                ROI_tensor = torch.unsqueeze(ROI_tensor, 0)
                output = SMILENet(ROI_tensor)
                pred = int(torch.max(output,1)[1][0].to(torch.uint8))
                """

    return OUT, (X_bot, X_top, Y_bot, Y_top)