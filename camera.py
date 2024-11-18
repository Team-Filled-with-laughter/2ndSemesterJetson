from flask import Flask, Response, request, jsonify
from flask_cors import CORS  # CORS를 import 합니다
import cv2
import random
import time

app = Flask(__name__)

# React 앱의 출처만 허용하고, credentials: 'include'도 허용
CORS(app, origins="https://hilarious-lollipop-a465ac.netlify.app", supports_credentials=True)

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

def event_stream():
    while True:
        # 임의의 값으로 v1, v2 설정 (예시)
        v1 = random.randint(0, 100)  # 임의의 값으로 v1 설정
        v2 = random.randint(0, 100)  # 임의의 값으로 v2 설정

        # 서버에서 v1, v2 값을 실시간으로 전송
        yield f"data: {{\"v1\": {v1}, \"v2\": {v2}}}\n\n".encode('utf-8')

        time.sleep(1)  # 1초마다 전송                  

# 비디오 스트림을 제공하는 라우트
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# v1, v2 값을 클라이언트로 실시간으로 전송하는 SSE 라우트
@app.route('/speed_feed')
def speed_feed():
    return Response(event_stream(), content_type='text/event-stream')

@app.route('/update_threshold', methods=['POST', 'OPTIONS'])
def update_threshold():
    if request.method == 'OPTIONS':
        # CORS preflight 요청 처리
        response = Response()
        # CORS 관련 헤더 추가
        response.headers['Access-Control-Allow-Origin'] = 'https://hilarious-lollipop-a465ac.netlify.app'  # React 앱의 출처만 허용
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'  # 허용할 메서드
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'  # 허용할 헤더
        response.headers['Access-Control-Allow-Credentials'] = 'true'  # 쿠키 포함 허용
        return response, 200  # 200 OK 응답

    # POST 요청 처리
    data = request.get_json()  # JSON 데이터 받기
    threshold = data.get('threshold')

    if threshold is not None:
        return jsonify({"message": "임계값 업데이트 성공", "threshold": threshold}), 200
    else:
        return jsonify({"error": "임계값을 찾을 수 없습니다."}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)  # React 앱과 포트 충돌 피하려면 다른 포트 사용
