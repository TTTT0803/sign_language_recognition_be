from main import app, db, Dictionary  # Import từ file main của bạn

def seed_dictionary():
    # Danh sách dữ liệu bạn muốn thêm
    data_list = [
        {
            "word": "A", 
            "desc": "Nắm bàn tay lại, ngón cái áp sát cạnh ngón trỏ.", 
            "img": "/dictionary/A_test.jpg"
        },
        {
            "word": "B", 
            "desc": "Giơ thẳng 4 ngón tay, ngón cái gập vào lòng bàn tay.", 
            "img": "/dictionary/B_test.jpg"
        },
        {
            "word": "C", 
            "desc": "Cong các ngón tay tạo thành hình chữ C.", 
            "img": "/dictionary/C_test.jpg"
        },
        {
            "word": "D", 
            "desc": "Giơ ngón trỏ thẳng lên, các ngón còn lại chạm vào ngón cái tạo vòng tròn.", 
            "img": "/dictionary/D_test.jpg"
        },
        {
            "word": "Hello", 
            "desc": "Đưa tay lên trán và vẫy nhẹ ra xa (giống kiểu chào quân đội nhưng mềm mại hơn).", 
            "img": "/dictionary/H_test.jpg"
        },
        {
            "word": "L", 
            "desc": "Hai tay nắm lại đan chéo trước ngực (ôm tim).", 
            "img": "/dictionary/L_test.jpg"
        }
    ]

    # Bắt buộc phải dùng app_context để làm việc với DB bên ngoài luồng request
    with app.app_context():
        print("🔄 Đang kiểm tra và thêm dữ liệu...")
        added_count = 0
        
        for item in data_list:
            # Kiểm tra trùng lặp
            exists = Dictionary.query.filter_by(word=item["word"]).first()
            
            if not exists:
                new_word = Dictionary(
                    word=item["word"],
                    description=item["desc"],
                    image_url=item["img"]
                )
                db.session.add(new_word)
                added_count += 1
                print(f"   + Đã thêm từ: {item['word']}")
            else:
                print(f"   - Từ '{item['word']}' đã có, bỏ qua.")
        
        db.session.commit()
        print(f"✅ Hoàn tất! Đã thêm mới {added_count} từ vào Database.")

if __name__ == "__main__":
    seed_dictionary()