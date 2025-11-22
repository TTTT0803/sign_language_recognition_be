from flask import Flask, Response, jsonify # <--- 1. Thêm jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2

app = Flask(__name__)
CORS(app)

model = YOLO('best (3).pt')

# 2. Tạo biến toàn cục để lưu chữ cái đang nhận diện
current_result = {
    "text": "...",
    "confidence": 0.0
}

def generate_frames():
    # Dùng từ khóa global để cập nhật biến bên ngoài
    global current_result 
    
    camera = cv2.VideoCapture(0) 
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        results = model(frame, conf=0.5) # Chỉ lấy độ tin cậy > 50%
        
        # --- 3. Lấy chữ cái ra khỏi kết quả ---
        if len(results[0].boxes) > 0:
            # Lấy box có độ tin cậy cao nhất
            box = results[0].boxes[0] 
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Lấy tên class (ví dụ: "A", "B", "Hello")
            detected_text = model.names[class_id]
            
            # Cập nhật vào biến toàn cục
            current_result = {
                "text": detected_text,
                "confidence": round(conf, 2)
            }
        else:
            # Nếu không thấy gì thì reset
            current_result = {
                "text": "...",
                "confidence": 0.0
            }
        # --------------------------------------

        annotated_frame = results[0].plot()
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- 4. API Mới: Để React gọi vào lấy chữ ---
@app.route('/api/status')
def get_status():
    return jsonify(current_result)

if __name__ == "__main__":
    print("Backend đang chạy...")
    app.run(host='0.0.0.0', port=5000, debug=True)