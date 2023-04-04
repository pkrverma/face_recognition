import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)

jobs_image = face_recognition.load_image_file("/home/ubuntu/Desktop/Souvik/steve-jobs.jpg")
jobs_encoding = face_recognition.face_encodings(jobs_image)[0]

ratan_tata_image = face_recognition.load_image_file("/home/ubuntu/Desktop/Souvik/ratan-tata.jpg")
ratan_tata_encoding = face_recognition.face_encodings(ratan_tata_image)[0]

carl_pei_image = face_recognition.load_image_file("/home/ubuntu/Desktop/Souvik/carl pei.jpg")
carl_pei_encoding = face_recognition.face_encodings(carl_pei_image)[0]

bezos_image = face_recognition.load_image_file("/home/ubuntu/Desktop/Souvik/bezos.jpg")
bezos_encoding = face_recognition.face_encodings(bezos_image)[0]

narendra_modi_image = face_recognition.load_image_file("/home/ubuntu/Desktop/Souvik/narendra -modi.jpg")
narendra_modi_encoding = face_recognition.face_encodings(narendra_modi_image)[0]

elonmusk_image = face_recognition.load_image_file("/home/ubuntu/Desktop/Souvik/elonmusk.jpg")
elonmusk_encoding = face_recognition.face_encodings(elonmusk_image)[0]

zuckerberg_image = face_recognition.load_image_file("/home/ubuntu/Desktop/Souvik/zuckerberg.jpg")
zuckerberg_encoding = face_recognition.face_encodings(zuckerberg_image)[0]

souvik_das_image = face_recognition.load_image_file("/home/ubuntu/Desktop/Souvik/souvik-das.jpg")
souvik_das_encoding = face_recognition.face_encodings(souvik_das_image)[0]

srk_image= face_recognition.load_image_file("/home/ubuntu/Desktop/Souvik/srk.jpg")
srk_encoding = face_recognition.face_encodings(srk_image)[0]


known_face_encoding = [
    jobs_encoding,
    ratan_tata_encoding,
    carl_pei_encoding,
    bezos_encoding,
    narendra_modi_encoding,
    elonmusk_encoding,
    zuckerberg_encoding,
    souvik_das_encoding,
    srk_encoding,
    # tesla_encoding,
    # souvik_encoding,

    # Nikola_tesla_encoding,
]

known_faces_names = [
    "jobs",
    "ratan tata",
    "carl pei",
    "bezos",
    "narendra modi",
    "elonmusk",
    "zuckerberg",
    "souvik",
    "srk",
    # "tesla",
    # "sadmona",
    # "Nikola tesla"
]

students = known_faces_names.copy()

face_locations = []
face_encodings = []
face_names = []
s = True


now = datetime.now()
current_date = now.strftime("%Y-%m-%d")


f = open(current_date+'.csv', 'w+', newline='')
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                known_face_encoding, face_encoding)
            name = ""
            face_distance = face_recognition.face_distance(
                known_face_encoding, face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_faces_names[best_match_index]

            face_names.append(name)
            if name in known_faces_names:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10, 100)
                fontScale = 1.5
                fontColor = (255, 0, 0)
                thickness = 3
                lineType = 2

                cv2.putText(frame, name+' Present',
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            thickness,
                            lineType)

                if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name, current_time])
    cv2.imshow("attendence system", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()
