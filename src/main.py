import cv2
import mediapipe as mp
import numpy as np
import math

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

#via canonical_face_model_uv_visualization.png
face_landmarks_lower = [78, 95, 88, 178, 87, 14, 317, 405, 318, 324, 308]
face_landmarks_upper = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]

def DrawSmile(arr_y, arr_x, col, img_shape):
    if (np.all(arr_x) > img_shape[0]) == False or (np.all(arr_y) > img_shape[1]) == False:
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


# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        success, image = cap.read()
        image = cv2.flip(image, 1)

        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        h, w, c = image.shape
        blank = np.zeros((h, w, c))

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
                
                DrawSmile(out_y_low, out_x_low, (0, 0, 255), (h, w))
                DrawSmile(out_y_up, out_x_up, (255, 0, 0), (h, w))
                
                    
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Face Mesh', image)
        cv2.imshow('Smile approx', blank)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()