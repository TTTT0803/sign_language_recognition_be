import threading
import time
import cv2
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from ultralytics import YOLO

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# --- CẤU HÌNH DB ---
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/sign_language_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- MODELS ---
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

# --- CẤU HÌNH AI & THREADING ---

# Load model
model = YOLO('best (3).pt')

# Biến toàn cục
global_frame = None
global_result = {"text": "...", "confidence": 0.0}
lock = threading.Lock()
camera_active = True

# --- 1. LUỒNG CAMERA (Đã giới hạn FPS cứng) ---
def camera_thread():
    global global_frame, camera_active
    cap = cv2.VideoCapture(0)
    
    # Giảm độ phân giải gốc của camera để nhẹ băng thông USB
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30) # Yêu cầu phần cứng chỉ gửi 30 fps
    
    while camera_active:
        start_time = time.time()
        
        success, frame = cap.read()
        if success:
            frame = cv2.flip(frame, 1)
            with lock:
                global_frame = frame.copy()
        
        # TÍNH TOÁN SLEEP: Đảm bảo chỉ chạy tối đa 30 FPS để tiết kiệm CPU
        # Nếu đọc ảnh quá nhanh, bắt nó ngủ bù
        process_time = time.time() - start_time
        if process_time < 0.033: # 1/30 = 0.033
            time.sleep(0.033 - process_time)
        else:
            time.sleep(0.01) # Ngủ tối thiểu để CPU thở

    cap.release()

# --- 2. LUỒNG AI (Giảm tải cực mạnh) ---
def ai_processing_thread():
    global global_result, camera_active
    
    while camera_active:
        img_for_ai = None
        
        # Lấy ảnh
        with lock:
            if global_frame is not None:
                # TỐI ƯU CỰC ĐOAN: Resize xuống rất nhỏ (256x192)
                # Kích thước này nhẹ hơn 640x480 gấp 6 lần
                img_for_ai = cv2.resize(global_frame, (256, 192))
        
        if img_for_ai is not None:
            # Chạy AI
            results = model(img_for_ai, conf=0.5, verbose=False)
            
            if len(results[0].boxes) > 0:
                box = results[0].boxes[0]
                detected_text = model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                
                global_result = {
                    "text": detected_text,
                    "confidence": round(conf, 2)
                }
            else:
                global_result = {"text": "...", "confidence": 0.0}
        
        # QUAN TRỌNG NHẤT: Sleep 0.4 giây
        # Nghĩa là AI chỉ chạy 2.5 lần/giây thay vì chạy liên tục.
        # CPU sẽ rảnh rỗi 90% thời gian.
        time.sleep(0.4) 

# Khởi động thread
t_cam = threading.Thread(target=camera_thread, daemon=True)
t_ai = threading.Thread(target=ai_processing_thread, daemon=True)
t_cam.start()
t_ai.start()

# --- FLASK ROUTES ---

def generate_frames():
    while True:
        # Giới hạn tốc độ gửi ảnh về trình duyệt (khoảng 20 FPS là đủ xem mượt)
        time.sleep(0.05) 
        
        frame_display = None
        with lock:
            if global_frame is not None:
                frame_display = global_frame.copy()
        
        if frame_display is None:
            continue
            
        # Vẽ kết quả (Lấy từ biến toàn cục đã được AI cập nhật chậm rãi)
        text = global_result["text"]
        conf = global_result["confidence"]
        
        if text != "...":
            cv2.putText(frame_display, f"{text} ({conf})", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Giảm chất lượng ảnh JPG xuống 70% để gửi qua mạng nhanh hơn, nhẹ trình duyệt hơn
        ret, buffer = cv2.imencode('.jpg', frame_display, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def get_status():
    return jsonify(global_result)

# --- CÁC API KHÁC (USER, DICTIONARY, HISTORY) GIỮ NGUYÊN ---
@app.route('/api/dictionary', methods=['GET'])
def get_dictionary():
    words = Dictionary.query.all()
    return jsonify([{'id': w.id, 'word': w.word, 'description': w.description, 'image_url': w.image_url} for w in words])

@app.route('/api/save_history', methods=['POST'])
def save_history():
    data = request.json
    db.session.add(History(detected_word=data.get('word'), confidence=data.get('confidence'), user_id=data.get('user_id')))
    db.session.commit()
    return jsonify({'message': 'OK'})

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    user = User.query.filter_by(username=data['email']).first()
    if user and user.password == data['password']:
        return jsonify({'success': True, 'user': {'id': user.id, 'email': user.username}})
    return jsonify({'success': False})

@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    if User.query.filter_by(username=data['email']).first(): return jsonify({'success': False})
    db.session.add(User(username=data['email'], password=data['password']))
    db.session.commit()
    return jsonify({'success': True})

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    # Tắt hoàn toàn debug mode để Flask chạy nhanh nhất có thể
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)