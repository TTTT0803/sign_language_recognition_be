import threading
import time
import cv2
import math
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from ultralytics import YOLO

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# --- 1. CẤU HÌNH DATABASE (GIỮ NGUYÊN) ---
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/sign_language_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- MODELS (GIỮ NGUYÊN) ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

class Dictionary(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    word = db.Column(db.String(10), nullable=False)
    description = db.Column(db.String(200))
    image_url = db.Column(db.String(200))

class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    detected_word = db.Column(db.String(10))
    confidence = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# --- 2. HỆ THỐNG AI & CAMERA ĐA LUỒNG (TRÁI TIM CỦA TỐI ƯU) ---

# Load model
model = YOLO('best (3).pt')

# Biến toàn cục dùng chung giữa các luồng
global_frame = None            # Lưu khung hình mới nhất từ camera
global_result = {              # Lưu kết quả AI mới nhất
    "text": "...",
    "confidence": 0.0
}
lock = threading.Lock()        # Khóa an toàn để tránh xung đột dữ liệu
camera_active = True           # Cờ kiểm soát camera

# --- LUỒNG CAMERA (Chỉ đọc ảnh, không chạy AI) ---
def camera_thread():
    global global_frame, camera_active
    # Mở camera
    cap = cv2.VideoCapture(0)
    # Giảm resolution đầu vào camera xuống mức trung bình để nhẹ máy
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while camera_active:
        success, frame = cap.read()
        if success:
            frame = cv2.flip(frame, 1) # Lật ảnh
            
            # Dùng lock để cập nhật ảnh an toàn
            with lock:
                global_frame = frame.copy()
        
        # Ngủ cực ngắn để giảm tải CPU không cần thiết (khoảng 30 FPS)
        time.sleep(0.015)
    
    cap.release()

# --- LUỒNG AI (Chạy ngầm, xử lý ảnh nhỏ) ---
def ai_processing_thread():
    global global_result, camera_active
    
    while camera_active:
        img_for_ai = None
        
        # 1. Lấy ảnh mới nhất từ luồng Camera
        with lock:
            if global_frame is not None:
                # TỐI ƯU QUAN TRỌNG: Resize ảnh xuống thật nhỏ chỉ để cho AI nhìn
                # 320x240 là đủ để nhận diện tay, giúp tốc độ tăng gấp 4 lần
                img_for_ai = cv2.resize(global_frame, (320, 240))
        
        if img_for_ai is not None:
            # 2. Chạy AI trên ảnh nhỏ
            results = model(img_for_ai, conf=0.5, verbose=False)
            
            # 3. Cập nhật kết quả
            if len(results[0].boxes) > 0:
                box = results[0].boxes[0]
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = model.names[class_id]
                
                global_result = {
                    "text": name,
                    "confidence": round(conf, 2)
                }
            else:
                global_result = {
                    "text": "...",
                    "confidence": 0.0
                }
        
        # TỐI ƯU: Chỉ chạy AI khoảng 10 lần/giây (0.1s nghỉ)
        # Để dành CPU cho việc truyền hình ảnh mượt mà
        time.sleep(0.1)

# Khởi chạy các luồng ngay khi App bắt đầu
# Lưu ý: daemon=True để luồng tự tắt khi tắt chương trình chính
t_cam = threading.Thread(target=camera_thread, daemon=True)
t_ai = threading.Thread(target=ai_processing_thread, daemon=True)
t_cam.start()
t_ai.start()

# --- 3. CÁC API FLASK ---

def generate_frames():
    """Hàm này chỉ lấy ảnh từ bộ nhớ và gửi đi, KHÔNG xử lý gì cả -> Rất nhanh"""
    while True:
        frame_display = None
        
        with lock:
            if global_frame is not None:
                frame_display = global_frame.copy()
        
        if frame_display is None:
            time.sleep(0.1)
            continue
            
        # Vẽ kết quả hiện tại lên hình (để người dùng thấy trực tiếp trên video)
        # Lấy kết quả từ biến toàn cục (do luồng AI cập nhật)
        text = global_result["text"]
        conf = global_result["confidence"]
        
        if text != "...":
            cv2.putText(frame_display, f"{text} ({conf})", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Encode sang JPG
        ret, buffer = cv2.imencode('.jpg', frame_display)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def get_status():
    return jsonify(global_result)

# --- CÁC API DB (GIỮ NGUYÊN) ---
@app.route('/api/dictionary', methods=['GET'])
def get_dictionary():
    words = Dictionary.query.all()
    output = [{'id': w.id, 'word': w.word, 'description': w.description, 'image_url': w.image_url} for w in words]
    return jsonify(output)

@app.route('/api/save_history', methods=['POST'])
def save_history():
    data = request.json
    new_record = History(detected_word=data.get('word'), confidence=data.get('confidence'), user_id=data.get('user_id'))
    db.session.add(new_record)
    db.session.commit()
    return jsonify({'message': 'Lưu thành công!'})

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    user = User.query.filter_by(username=data['email']).first()
    if user and user.password == data['password']:
        return jsonify({'success': True, 'message': 'OK', 'user': {'id': user.id, 'email': user.username}})
    return jsonify({'success': False, 'message': 'Fail'})

@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    if User.query.filter_by(username=data['email']).first():
        return jsonify({'success': False, 'message': 'Email tồn tại'})
    new_user = User(username=data['email'], password=data['password'])
    db.session.add(new_user)
    db.session.commit()
    return jsonify({'success': True})

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    print("Backend AI Multi-thread đang chạy...")
    # use_reloader=False để tránh chạy 2 lần thread camera gây lỗi
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)