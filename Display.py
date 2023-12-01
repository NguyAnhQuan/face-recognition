import cv2
import tensorflow as tf
import numpy as np

id = 2

if id == 1: #nhận diện thông qua ảnh

    filename = r'D:\test face detect\Code main\Code\test\test3.jpg'
    image = cv2.imread(filename)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    save_model = tf.keras.models.load_model("khuonmat.h5")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fa = face_cascade.detectMultiScale(gray, 1.1, 5)
    fontface = cv2.FONT_HERSHEY_SIMPLEX
    for (x, y, w, h) in fa:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(src=roi_gray, dsize=(100, 100))
        roi_gray = roi_gray.reshape((100, 100, 1))
        roi_gray = np.array(roi_gray)
        result = save_model.predict(np.array([roi_gray]))
        final = np.argmax(result)
        confidence = result[0][final] * 100
        if confidence < 60:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(image, "Unknown", (x + 10, y + h + 30), fontface, 1, (0, 0, 255), 2)
        else:
            if final == 0:
                cv2.putText(image, "Hai trieu drop test", (x + 10, y + h + 30), fontface, 1, (0, 255, 0), 2)
            elif final == 1:
                cv2.putText(image, "Son tung MTP", (x + 10, y + h + 30), fontface, 1, (0, 255, 0), 2)
            elif final == 2:
                cv2.putText(image, "Jenny Huynh", (x + 10, y + h + 30), fontface, 1, (0, 255, 0), 2)
            elif final == 3:
                cv2.putText(image, "Pham Nhat Vuong", (x + 10, y + h + 30), fontface, 1, (0, 255, 0), 2)
            elif final == 4:
                cv2.putText(image, "Sark Linh", (x + 10, y + h + 30), fontface, 1, (0, 255, 0), 2)
        
        cv2.putText(image, f"Confidence: {confidence:.2f}%", (x + 10, y + h + 60), fontface, 1, (0, 255, 0), 2)
    
    # Tính toán kích thước mới (50% kích thước ảnh gốc)
    new_width = int(image.shape[1] * 0.7)
    new_height = int(image.shape[0] * 0.7)

    # Thay đổi kích thước hình ảnh
    resized_image = cv2.resize(image, (new_width, new_height))

    # Hiển thị hình ảnh đã thay đổi kích thước trong cửa sổ
    cv2.imshow('training', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if id == 2: # dùng trực tiếp bằng camera
    # Khởi tạo camera
    cap = cv2.VideoCapture(0)  # Số 0 thường là camera mặc định, nếu bạn có nhiều camera, bạn có thể thay đổi số này.

    # Load mô hình và khai báo biến cần thiết
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    save_model = tf.keras.models.load_model("khuonmat.h5")
    fontface = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, frame = cap.read()  # Đọc hình ảnh từ camera

        if not ret:
            break

        # Chuyển hình ảnh sang ảnh xám
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Phát hiện khuôn mặt
        fa = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in fa:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(src=roi_gray, dsize=(100, 100))
            roi_gray = roi_gray.reshape((100, 100, 1))
            roi_gray = np.array(roi_gray)
            result = save_model.predict(np.array([roi_gray]))
            final = np.argmax(result)
            confidence = result[0][final] * 99
            if confidence < 90:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown", (x + 10, y + h + 30), fontface, 1, (0, 0, 255), 2)
            else:
                if final == 0:
                    cv2.putText(frame, "Hai trieu drop test", (x + 10, y + h + 30), fontface, 1, (0, 255, 0), 2)
                elif final == 1:
                    cv2.putText(frame, "Son tung MTP", (x + 10, y + h + 30), fontface, 1, (0, 255, 0), 2)
                elif final == 2:
                    cv2.putText(frame, "Jenny Huynh", (x + 10, y + h + 30), fontface, 1, (0, 255, 0), 2)
                elif final == 3:
                    cv2.putText(frame, "Pham Nhat Vuong", (x + 10, y + h + 30), fontface, 1, (0, 255, 0), 2)
                elif final == 4:
                    cv2.putText(frame, "Sark Linh", (x + 10, y + h + 30), fontface, 1, (0, 255, 0), 2)
            
            cv2.putText(frame, f"Confidence: {confidence:.2f}%", (x + 10, y + h + 60), fontface, 1, (0, 255, 0), 2)

        cv2.imshow('Camera', frame)

        # Để thoát vòng lặp, bấm phím 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng tài nguyên và đóng cửa sổ
    cap.release()
    cv2.destroyAllWindows()
