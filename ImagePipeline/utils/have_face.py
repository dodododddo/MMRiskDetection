import face_recognition

def have_face(image_path):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image) 
    amount = len(face_locations)
    if amount != 0:
        return True
    else:
        return False

