import cv2
from flask import Flask, render_template, Response
from flask_cors import CORS  # CORS를 import 합니다

app = Flask(__name__)
CORS(app)  # CORS 설정

# OpenCV로 카메라 스트림 열기
cap = cv2.VideoCapture(0)  # 0번 카메라는 기본 카메라 (웹캠)

# 카메라 프레임을 JPEG 형식으로 인코딩하여 웹에서 받을 수 있도록 함
def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# 비디오 스트림을 제공하는 라우트
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)  # React 앱과 포트 충돌 피하려면 다른 포트 사용
