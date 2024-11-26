import cv2
from flask import Flask, Response, request, jsonify
import random
import time
from flask_cors import CORS

app = Flask(__name__)

# 비디오 파일 경로 설정 (test_video.mp4 또는 다른 비디오 파일 경로)
video_source = 'test_video.mp4'  # 비디오 파일 경로
CORS(app, supports_credentials=False)


def event_stream():
    while True:
        # 임의의 값으로 v1, v2 설정 (예시)
        v1 = random.randint(0, 100)  # 임의의 값으로 v1 설정
        v2 = random.randint(0, 100)  # 임의의 값으로 v2 설정

        # 서버에서 v1, v2 값을 실시간으로 전송
        yield f"data: {{\"v1\": {v1}, \"v2\": {v2}}}\n\n".encode('utf-8')

        time.sleep(1)  # 1초마다 전송


def generate_frames():
    cap = cv2.VideoCapture(video_source)

    # 비디오 해상도 낮추기 (예: 640x480으로 설정)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # 프레임 크기 줄이기 (저해상도 영상으로 처리 속도 향상)
            frame = cv2.resize(frame, (640, 480))

            # MJPEG로 인코딩
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame = buffer.tobytes()

            # MJPEG 스트리밍 형식에 맞춰 전송
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()


@app.route('/speed_feed')
def speed_feed():
    return Response(event_stream(), content_type='text/event-stream')


@app.route('/video_feed')
def video_feed():
    # 비디오 스트리밍을 제공하는 경로
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/update_actuator', methods=['POST', 'OPTIONS'])
def update_actuator():
    # CORS preflight 요청 처리
    if request.method == 'OPTIONS':
        response = Response()
        origin = request.headers.get('Origin')
        allowed_origins = ['http://13.209.45.118:8080']  # List of allowed origins

        if origin in allowed_origins:
            response.headers['Access-Control-Allow-Origin'] = origin
        else:
            response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'  # 허용할 메서드
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'  # 허용할 헤더
        return response, 200  # 200 OK 응답

    # POST 요청 처리
    data = request.get_json()  # JSON 데이터 받기
    up = data.get('up')

    if up is not None:
        if up == 1:
            print('Up success')
            return jsonify({"message": "엑추에이터 상승"}), 200
        else:
            print('Down success')
            return jsonify({"message": "엑추에이터 하강"}), 200
    else:
        return jsonify({"error": "엑추에이터 버튼 오류."}), 400


@app.route('/update_threshold', methods=['POST', 'OPTIONS'])
def update_threshold():
    if request.method == 'OPTIONS':
        # CORS preflight 요청 처리
        response = Response()
        # CORS 관련 헤더 추가
        origin = request.headers.get('Origin')
        allowed_origins = ['http://ec2-13-209-45-118.ap-northeast-2.compute.amazonaws.com:8080']  # List of allowed origins

        if origin in allowed_origins:
            response.headers['Access-Control-Allow-Origin'] = origin
        else:
            response.headers['Access-Control-Allow-Origin'] = '*'  # or ret
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'  # 허용할 메서드
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'  # 허용할 헤더
        return response, 200  # 200 OK 응답




    data = request.get_json()  # JSON 데이터 받기
    threshold = data.get('threshold')

    if threshold is not None:
        print("임계값: ", threshold)
        return jsonify({"message": "임계값 업데이트 성공", "threshold": threshold}), 200
    else:
        return jsonify({"error": "임계값을 찾을 수 없습니다."}), 400


if __name__ == '__main__':
    app.run( debug=True, host='0.0.0.0', port=5001)
