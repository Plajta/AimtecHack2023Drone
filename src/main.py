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

def JawComputations(jaw_arr):

    jaw_arr_y = jaw_arr[:, 0]
    jaw_arr_x = jaw_arr[:, 1]

    y_min = np.amin(jaw_arr_y)
    x_min = np.amin(jaw_arr_x)

    y_max = np.amax(jaw_arr_y)
    x_max = np.amax(jaw_arr_x)

    #normalize data
    jaw_arr_y = jaw_arr_y - y_min
    jaw_arr_x = jaw_arr_x - x_min

    #fit polynomial model
    poly_model = np.poly1d(np.polyfit(jaw_arr_x, jaw_arr_y, 3))
    
    #apply polynomial model on y_min ... to ... y_max space
    x_len = x_max - x_min
    
    space = np.arange(x_len)
    y_space = np.floor(poly_model(space))

    out_x = space + x_min
    out_y = y_space + y_min

    return out_y, out_x


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
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        h, w, c = image.shape

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
                    X = face_landmarks.landmark[i_low].x
                    Y = face_landmarks.landmark[i_low].y

                    X = math.floor(X * w)
                    Y = math.floor(Y * h)

                    jaw_up[i, 0] = Y
                    jaw_up[i, 1] = X

                out_y_low, out_x_low = JawComputations(jaw_low)
                out_y_up, out_y_up = JawComputations(jaw_up)

                #JawComputations(jaw_low)

                """
                for i in range(out_x_low.shape[0]):
                    cv2.circle(image, (out_x_low[i], out_y_low[i]), radius=0, color=(0, 0, 255), thickness=-1)
                for i in range(out_y_low.shape[0]):
                    cv2.circle(image, (out_x_low[i], out_y_low[i]), radius=0, color=(0, 0, 255), thickness=-1)
                """
                    
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()