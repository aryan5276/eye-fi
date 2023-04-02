import math
from flask import Flask, redirect, render_template, request, url_for, session
from twilio.rest import Client
from pymongo import MongoClient
from imutils import face_utils
from utils import *
import numpy as np
import pyautogui as pyag
import imutils
import dlib
import cv2
import numpy as np
from multiprocessing import Process
import face_recognition
import pickle
import sys
import time

# twilio api credentials
account_sid = "AC19c5185f5a78f8792fe1b5bcb64db168"
auth_token = "9c905d02b7b842ffed2b38707e83e409"
client = Client(account_sid, auth_token)

# mongodb connections
cluster = "mongodb+srv://aryan527:1234@cluster0.p84jhqr.mongodb.net/?retryWrites=true&w=majority"
mongo_client = MongoClient(cluster)
db = mongo_client.userInfo
userInfo = db.userInfo


# flask connection
app = Flask(__name__)
app.secret_key = "Hehe1234"


# Global var to store name
g_name = ""


@app.route("/")
def index(): 
    return render_template("index.html")


# @app.route("/login")
# def login(): 
#     status = request.args.get("status")
#     name = request.args.get("name")
#     phone_number = request.args.get("phone_number")
#     return render_template("login.html", phone = phone_number, status = status, name = name)

@app.route("/face_auth", methods=("GET", "POST"))
def face_auth():
    if request.method == "POST":
        print(1)
        code = start_sign_in()
        
        print(2)
        
        status = 'Approved' if code == 0 else 'Try Again'

        session["login-status"] = status
        # session["old-name"] = name

        return render_template("face_auth.html", name = None, status = status)


    status = request.args.get("status")
    name = request.args.get("name")
    return render_template("face_auth.html", name = name, status = status)


@app.route("/verifyface", methods=("GET", "POST"))
def verifyface():
    if request.method == "POST":
        name = request.form["logname"]
        sign_up(name)
        status = request.args.get("status")
        session["old-name"] = name
        return render_template("face_auth.html", name = name, status = status)


    status = request.args.get("status")
    name = request.args.get("name")
    return render_template("face_auth.html", name = name, status = status)

@app.route("/login", methods=("GET", "POST"))
def login():
    if request.method == "POST":
        name = request.form["logname"]
        phone_number = request.form["logphone"]
        session["old-phone"] = phone_number
        session["old-name"] = name
        
        status = request.args.get("status")
        
        print(send_otp(phone_number))

        return render_template("login.html", phone = phone_number, status = status, name = name)        


    status = request.args.get("status")
    name = request.args.get("name")
    phone_number = request.args.get("phone_number")
    

    # render the third page
    return render_template("login.html", phone = phone_number, status = status, name = name)


@app.route("/verifyform", methods=("GET", "POST"))
def verifyform():
    if request.method == "POST":
        otp = request.form["logpass"]
        
        phone = session.get("old-phone")
        name = session.get("old-name")

        
        status = verify_otp(phone, otp) if phone != None else 0
        status = 'Approved' if status == 'approved' else 'Try Again'
        session["login-status"] = status

        # check if already a user (MongoDB Atlas)
        exists = False
        res = userInfo.find({"phone": phone})
        for r in res:
            if r["phone"] == phone:
                exists = True
                
        # add if doesnt exist
        if not exists and status == 'Approved':
            data = {"phone": phone, "name": name, "nav": 1}
            userInfo.insert_one(data)

        session["login-status"] = status
        return render_template("login.html", phone = phone, status = status, name = name)

    phone = request.args.get("phone")
    session["old-phone"] = phone
    status = request.args.get("status")
    name = request.args.get("name")
    session["old-name"] = name
    return render_template("login.html", phone = phone, status = status, name = name)


@app.route("/nav", methods = ("GET", "POST"))
def nav():
    if request.method == "POST":
        status = session.get("login-status")
        nav = request.args.get("nav")




        return render_template("nav", status = status)

    # nav = (userInfo.find_one({"phone": session.get("old-phone")}))["nav"]
    status = session.get("login-status")

    return render_template("nav.html", status = status)


@app.route('/process_checkbox', methods=['POST'])
def process_checkbox():
    data = request.json
    checkbox_value = data['value']
    # do something with the checkbox value
    if checkbox_value:
        start_cursor()

    return render_template("nav.html", status=None)




def send_otp(phone_number):
    verification = client.verify \
                     .v2 \
                     .services('VA59de1d735cbf39c803a1f60bf9f46928') \
                     .verifications \
                     .create(
                          to='+1' + str(phone_number),
                          channel='sms'
                      )

    v_sid = verification.sid
    return v_sid


def verify_otp(phone_number, code):
    verification_check = client.verify \
                           .v2 \
                           .services('VA59de1d735cbf39c803a1f60bf9f46928') \
                           .verification_checks \
                           .create(to='+1' + str(phone_number), code=code)

    return verification_check.status


def sign_in():
    cap = cv2.VideoCapture(0)
    try:
        with open('face_data.dat', 'rb') as f:
            known_face_data = pickle.load(f)
        known_face_encodings = known_face_data['encodings']
        known_face_labels = known_face_data['labels']
    except FileNotFoundError:
        known_face_encodings = []
        known_face_labels = []

    # Capture frame-by-frame
    ret, frame = cap.read()

    # Find all the faces and their encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Loop through each face found in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare the current face encoding with the known face encodings
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        # If a match is found, display a welcome message and stop capturing video
        if True in matches:
            label = known_face_labels[matches.index(True)]
            print("Authentication successful! Welcome, " + label + "!")
            cv2.putText(frame, "Authentication successful! Welcome, " + label + "!", (left-10, bottom+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cap.release()
            cv2.destroyAllWindows()
            return 1
            break

    cap.release()
    cv2.destroyAllWindows()
    sys.exit(1)
    



def sign_up(name):
    cap = cv2.VideoCapture(0)
    try:
        with open('face_data.dat', 'rb') as f:
            known_face_data = pickle.load(f)
        known_face_encodings = known_face_data['encodings']
        known_face_labels = known_face_data['labels']
    except FileNotFoundError:
        known_face_encodings = []
        known_face_labels = []

    ret, frame = cap.read()

        # Find all the faces and their encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

        # Prompt user to enter a label for the face
    if len(face_encodings) > 0:
        # cv2.imshow('Video', frame)
        label = name
        known_face_encodings.append(face_encodings[0])
        known_face_labels.append(label)

            # Save known face encodings and corresponding labels to file
        with open('face_data.dat', 'wb') as f:
            pickle.dump({'encodings': known_face_encodings, 'labels': known_face_labels}, f)


# Verbwire
def mint_custom_nft(data, username):
    url = "https://api.verbwire.com/v1/nft/mint/mintFromMetadata"
    
    payload = f"""\
-----011000010111000001101001\r
Content-Disposition: form-data; name="quantity"\r
\r
1\r
-----011000010111000001101001\r
Content-Disposition: form-data; name="chain"\r
\r
goerli\r
-----011000010111000001101001\r
Content-Disposition: form-data; name="contractAddress"\r
\r
0xc83E1Dad8fC1A872420154dFbb5b318aaf769940\r
-----011000010111000001101001\r
Content-Disposition: form-data; name="data"\r
\r
{data}\r
-----011000010111000001101001\r
Content-Disposition: form-data; name="recipientAddress"\r
\r
0x717aeB89048f10061C0dCcdEB2592a60bA4F1a79\r
-----011000010111000001101001\r
Content-Disposition: form-data; name="name"\r
\r
{username}\r
-----011000010111000001101001--\r
"""
    headers = {
        "accept": "application/json",
        "content-type": "multipart/form-data; boundary=---011000010111000001101001",
        "X-API-Key": "sk_live_b7159a98-601c-455e-b0e8-fd8cb42b48b3"
    }
    
    response = requests.post(url, data=payload, headers=headers)
    
    return response.text

# Verbwire
def get_nft_attributes():
    url = "https://api.verbwire.com/v1/nft/data/owned?walletAddress=0x717aeB89048f10061C0dCcdEB2592a60bA4F1a79&chain=goerli"
    headers = {
        "accept": "application/json",
        "X-API-Key": "sk_live_b7159a98-601c-455e-b0e8-fd8cb42b48b3"
    }
    response = requests.get(url, headers=headers)
    
    data = response.json()
    token_attributes = []

    for nft in data['nfts']:
        contract_address = nft['contractAddress']
        if contract_address == "0xc83E1Dad8fC1A872420154dFbb5b318aaf769940".lower():
            token_id = nft["tokenID"]
            chain = "goerli"
            url_inner = f"https://api.verbwire.com/v1/nft/data/nftDetails?contractAddress={contract_address}&tokenId={token_id}&chain={chain}"
            headers = {
                "accept": "application/json",
                "X-API-Key": "sk_live_b7159a98-601c-455e-b0e8-fd8cb42b48b3"
            }
            url_resp = requests.get(url_inner, headers=headers)
            json_data = url_resp.json()
            token_uri = json_data['nft_details']['tokenURI']
            main_resp = requests.get(token_uri).json()
            token_attributes.appends(main_resp['attributes'])
    return token_attributes


# Thread generation
def start_cursor():
    cursor_process = Process(target=cursor)
    cursor_process.start()


def start_sign_up(name):
    signup_process = Process(target=sign_up, args=(name,))
    signup_process.start()

def start_sign_in():
    sign_in_process = Process(target=sign_in)
    sign_in_process.start()
    sign_in_process.join()
    return sign_in_process.exitcode

# OpenCV
def cursor():
    # Returns EAR given eye landmarks
    def eye_aspect_ratio(eye):
        # Compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])

        # Compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = np.linalg.norm(eye[0] - eye[3])

        # Compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)

        # Return the eye aspect ratio
        return ear


    # Returns MAR given eye landmarks
    def mouth_aspect_ratio(mouth):
        # Compute the euclidean distances between the three sets
        # of vertical mouth landmarks (x, y)-coordinates
        A = np.linalg.norm(mouth[13] - mouth[19])
        B = np.linalg.norm(mouth[14] - mouth[18])
        C = np.linalg.norm(mouth[15] - mouth[17])

        # Compute the euclidean distance between the horizontal
        # mouth landmarks (x, y)-coordinates
        D = np.linalg.norm(mouth[12] - mouth[16])

        # Compute the mouth aspect ratio
        mar = (A + B + C) / (2 * D)

        # Return the mouth aspect ratio
        return mar


    # Return direction given the nose and anchor points.
    def direction(nose_point, anchor_point, w, h, multiple=1):
        nx, ny = nose_point
        x, y = anchor_point

        if nx > x + multiple * w:
            return 'right'
        elif nx < x - multiple * w:
            return 'left'

        if ny > y + multiple * h:
            return 'down'
        elif ny < y - multiple * h:
            return 'up'

        return 'none'
    # Thresholds and consecutive frame length for triggering the mouse action.
    MOUTH_AR_THRESH = 0.6
    MOUTH_AR_CONSECUTIVE_FRAMES = 15
    EYE_AR_THRESH = 0.19
    EYE_AR_CONSECUTIVE_FRAMES = 15
    WINK_AR_DIFF_THRESH = 0.04
    WINK_AR_CLOSE_THRESH = 0.19
    WINK_CONSECUTIVE_FRAMES = 3

    # Initialize the frame counters for each action as well as 
    # booleans used to indicate if action is performed or not
    MOUTH_COUNTER = 0
    EYE_COUNTER = 0
    WINK_COUNTER = 0
    INPUT_MODE = False
    EYE_CLICK = False
    LEFT_WINK = False
    RIGHT_WINK = False
    SCROLL_MODE = False
    ANCHOR_POINT = (0, 0)
    WHITE_COLOR = (255, 255, 255)
    YELLOW_COLOR = (0, 255, 255)
    RED_COLOR = (0, 0, 255)
    GREEN_COLOR = (0, 255, 0)
    BLUE_COLOR = (255, 0, 0)
    BLACK_COLOR = (0, 0, 0)

    # Initialize Dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor.dat')

    # Grab the indexes of the facial landmarks for the left and
    # right eye, nose and mouth respectively
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    # Video capture
    vid = cv2.VideoCapture(0)
    resolution_w = 1366
    resolution_h = 768
    cam_w = 640
    cam_h = 480
    unit_w = resolution_w / cam_w
    unit_h = resolution_h / cam_h

    while True:
        # Grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale
        # channels)
        _, frame = vid.read()
        frame = cv2.flip(frame, 1)
        frame = imutils.resize(frame, width=cam_w, height=cam_h)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        rects = detector(gray, 0)

        # Loop over the face detections
        if len(rects) > 0:
            rect = rects[0]
        else:
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            continue

        # Determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        mouth = shape[mStart:mEnd]
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        nose = shape[nStart:nEnd]

        # Because I flipped the frame, left is right, right is left.
        temp = leftEye
        leftEye = rightEye
        rightEye = temp

        # Average the mouth aspect ratio together for both eyes
        mar = mouth_aspect_ratio(mouth)
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        diff_ear = np.abs(leftEAR - rightEAR)

        nose_point = (nose[3, 0], nose[3, 1])

        # Compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        mouthHull = cv2.convexHull(mouth)
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [mouthHull], -1, YELLOW_COLOR, 1)
        cv2.drawContours(frame, [leftEyeHull], -1, YELLOW_COLOR, 1)
        cv2.drawContours(frame, [rightEyeHull], -1, YELLOW_COLOR, 1)

        for (x, y) in np.concatenate((mouth, leftEye, rightEye), axis=0):
            cv2.circle(frame, (x, y), 2, GREEN_COLOR, -1)
            
        # Check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if diff_ear > WINK_AR_DIFF_THRESH:

            if leftEAR < rightEAR:
                if leftEAR < EYE_AR_THRESH:
                    WINK_COUNTER += 1

                    if WINK_COUNTER > WINK_CONSECUTIVE_FRAMES:
                        pyag.click(button='left')

                        WINK_COUNTER = 0

            elif leftEAR > rightEAR:
                if rightEAR < EYE_AR_THRESH:
                    WINK_COUNTER += 1

                    if WINK_COUNTER > WINK_CONSECUTIVE_FRAMES:
                        pyag.click(button='right')

                        WINK_COUNTER = 0
            else:
                WINK_COUNTER = 0
        else:
            if ear <= EYE_AR_THRESH:
                EYE_COUNTER += 1

                if EYE_COUNTER > EYE_AR_CONSECUTIVE_FRAMES:
                    SCROLL_MODE = not SCROLL_MODE
                    # INPUT_MODE = not INPUT_MODE
                    EYE_COUNTER = 0

                    # nose point to draw a bounding box around it

            else:
                EYE_COUNTER = 0
                WINK_COUNTER = 0

        if mar > MOUTH_AR_THRESH:
            MOUTH_COUNTER += 1

            if MOUTH_COUNTER >= MOUTH_AR_CONSECUTIVE_FRAMES:
                # if the alarm is not on, turn it on
                INPUT_MODE = not INPUT_MODE
                # SCROLL_MODE = not SCROLL_MODE
                MOUTH_COUNTER = 0
                ANCHOR_POINT = nose_point

        else:
            MOUTH_COUNTER = 0

        if INPUT_MODE:
            cv2.putText(frame, "READING INPUT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2)
            x, y = ANCHOR_POINT
            nx, ny = nose_point
            w, h = 60, 35
            multiple = 1
            cv2.rectangle(frame, (x - w, y - h), (x + w, y + h), GREEN_COLOR, 2)
            cv2.line(frame, ANCHOR_POINT, nose_point, BLUE_COLOR, 2)

            dir = direction(nose_point, ANCHOR_POINT, w, h)
            cv2.putText(frame, dir.upper(), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2)
            drag = 18
            if dir == 'right':
                pyag.moveRel(drag, 0)
            elif dir == 'left':
                pyag.moveRel(-drag, 0)
            elif dir == 'up':
                if SCROLL_MODE:
                    pyag.scroll(40)
                else:
                    pyag.moveRel(0, -drag)
            elif dir == 'down':
                if SCROLL_MODE:
                    pyag.scroll(-40)
                else:
                    pyag.moveRel(0, drag)

        if SCROLL_MODE:
            cv2.putText(frame, 'SCROLL MODE IS ON!', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2)

        # cv2.putText(frame, "MAR: {:.2f}".format(mar), (500, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW_COLOR, 2)
        # cv2.putText(frame, "Right EAR: {:.2f}".format(rightEAR), (460, 80),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW_COLOR, 2)
        # cv2.putText(frame, "Left EAR: {:.2f}".format(leftEAR), (460, 130),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW_COLOR, 2)
        # cv2.putText(frame, "Diff EAR: {:.2f}".format(np.abs(leftEAR - rightEAR)), (460, 80),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # If the `Esc` key was pressed, break from the loop
        if key == 27:
            break

    # Do a bit of cleanup
    cv2.destroyAllWindows()
    vid.release()