import cv2
import face_recognition

# Load known face encodings and names
known_face_encodings = []
known_face_names = []

# Load and encode known faces
person1_image = face_recognition.load_image_file(r"D:\Computer science\Research\FacialRecognizor1\Person 1\Person_5.jpg")
person2_image = face_recognition.load_image_file(r"D:\Computer science\Research\FacialRecognizor1\Person 1\Person_1.jpg")
person3_image = face_recognition.load_image_file(r"D:\Computer science\Research\FacialRecognizor1\Person 1\Person_4.jpg")

person1_encoding = face_recognition.face_encodings(person1_image)[0]
person2_encoding = face_recognition.face_encodings(person2_image)[0]
person3_encoding = face_recognition.face_encodings(person3_image)[0]

# Append known encodings and names
known_face_encodings = [person1_encoding, person2_encoding, person3_encoding]
known_face_names = ["Arnold", "Ronaldo", "Messi"]

# Initialize webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture image")
        break

    # Find face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Loop through each face found in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the face matches any known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # If a match was found, use the first one
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw a rectangle around the face and label it
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow("Video", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
video_capture.release()
cv2.destroyAllWindows()
