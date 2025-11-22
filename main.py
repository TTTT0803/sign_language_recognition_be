from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy # Thư viện DB
from datetime import datetime
from ultralytics import YOLO
import cv2

app = Flask(__name__)
# Cho phép mọi nguồn kết nối (tránh lỗi CORS khi gọi từ React)
CORS(app, resources={r"/*": {"origins": "*"}})

# --- 1. CẤU HÌNH KẾT NỐI MYSQL (HEIDISQL) ---
# Cú pháp: mysql+pymysql://USERNAME:PASSWORD@HOST:PORT/TEN_DB
# Mặc định XAMPP: User='root', Pass='', Host='localhost', DB='sign_language_db'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/sign_language_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# --- 2. ĐỊNH NGHĨA CÁC BẢNG (MODELS) ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

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

# --- 3. PHẦN AI & CAMERA (GIỮ NGUYÊN CODE CỦA BẠN) ---
# Hãy đảm bảo file 'best (3).pt' nằm cùng thư mục với main.py
model = YOLO('best (3).pt') 

# Biến toàn cục để lưu chữ cái đang nhận diện
current_result = {
    "text": "...",
    "confidence": 0.0
}

def generate_frames():
    global current_result 
    camera = cv2.VideoCapture(0) 
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        results = model(frame, conf=0.5) 
        
        # Lấy chữ cái ra khỏi kết quả
        if len(results[0].boxes) > 0:
            box = results[0].boxes[0] 
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            detected_text = model.names[class_id]
            
            # Cập nhật vào biến toàn cục
            current_result = {
                "text": detected_text,
                "confidence": round(conf, 2)
            }
        else:
            current_result = {
                "text": "...",
                "confidence": 0.0
            }

        annotated_frame = results[0].plot()
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# --- 4. CÁC API CHO FRONTEND REACT GỌI VÀO ---

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def get_status():
    return jsonify(current_result)

# API: Lấy danh sách từ điển
@app.route('/api/dictionary', methods=['GET'])
def get_dictionary():
    words = Dictionary.query.all()
    output = []
    for w in words:
        output.append({
            'id': w.id,
            'word': w.word,
            'description': w.description,
            'image_url': w.image_url
        })
    return jsonify(output)

# API: Lưu lịch sử (Frontend gọi khi người dùng bấm nút "Lưu")
@app.route('/api/save_history', methods=['POST'])
def save_history():
    data = request.json
    new_record = History(
        detected_word=data.get('word'),
        confidence=data.get('confidence'),
        user_id=data.get('user_id') # Có thể null
    )
    db.session.add(new_record)
    db.session.commit()
    return jsonify({'message': 'Đã lưu vào Database thành công!'})

# API: Tạo dữ liệu mẫu (Chạy 1 lần để nạp A, B, C vào database)
@app.route('/init_data')
def init_data():
    if Dictionary.query.count() == 0:
        w1 = Dictionary(word="A", description="Nắm bàn tay lại, ngón cái áp sát ngón trỏ.", image_url="https://example.com/a.png")
        w2 = Dictionary(word="B", description="Giơ thẳng 4 ngón tay, ngón cái gập vào lòng bàn tay.", image_url="https://example.com/b.png")
        w3 = Dictionary(word="C", description="Cong các ngón tay tạo thành hình chữ C.", image_url="https://example.com/c.png")
        w4 = Dictionary(word="Hello", description="Đưa tay lên trán và vẫy ra xa.", image_url="https://example.com/hello.png")
        
        db.session.add_all([w1, w2, w3, w4])
        db.session.commit()
        return "Đã thêm dữ liệu mẫu (A, B, C, Hello) vào MySQL thành công!"
    return "Dữ liệu đã có sẵn trong MySQL."
# --- THÊM VÀO main.py ---

# API 5: Đăng ký tài khoản mới
@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    # Kiểm tra xem email đã tồn tại chưa
    existing_user = User.query.filter_by(username=data['email']).first()
    if existing_user:
        return jsonify({'success': False, 'message': 'Email này đã được sử dụng!'})
    
    # Tạo user mới
    new_user = User(
        username=data['email'],  # Dùng email làm username
        password=data['password'] # Lưu ý: Thực tế nên mã hóa password, ở đây làm demo nên để trần
    )
    db.session.add(new_user)
    db.session.commit()
    return jsonify({'success': True, 'message': 'Đăng ký thành công!'})

# API 6: Đăng nhập
@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    # Tìm user trong database
    user = User.query.filter_by(username=data['email']).first()
    
    # Kiểm tra mật khẩu
    if user and user.password == data['password']:
        return jsonify({
            'success': True, 
            'message': 'Đăng nhập thành công!',
            'user': {'id': user.id, 'email': user.username}
        })
    
    return jsonify({'success': False, 'message': 'Sai email hoặc mật khẩu!'})

if __name__ == "__main__":
    # Lệnh này tự động tạo các bảng trong MySQL nếu chưa có
    with app.app_context():
        db.create_all()
        print("Đã kết nối MySQL và khởi tạo bảng thành công!")

    print("Backend đang chạy...")
    app.run(host='0.0.0.0', port=5000, debug=True)