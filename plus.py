print('isOverSpeed: ', isOverSpeed)
print('speed :', speed)
print('v1 :', v1)
#####################################################
yield f"data: {{\"v1\": {v1}, \"v2\": {speed}}}\n\n"
######################################################
clear(listA)

@app.route('/speed_feed')
def speed_feed():
    return Response(actuatorPlay(), content_type='text/event-stream') # 이렇게하면 web실행할때만 될듯?

@app.route('/update_threshold', methods=['POST', 'OPTIONS'])
def update_threshold():
    # CORS preflight 요청 처리
    if request.method == 'OPTIONS':
        response = Response()
        response.headers['Access-Control-Allow-Origin'] = 'https://team-filled-with-laughter.netlify.app'  # React 앱의 출처만 허용
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'  # 허용할 메서드
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'  # 허용할 헤더
        response.headers['Access-Control-Allow-Credentials'] = 'true'  # 쿠키 포함 허용
        return response, 200  # 200 OK 응답

    # POST 요청 처리
    data = request.get_json()  # JSON 데이터 받기
    threshold = data.get('threshold')

    if threshold is not None:
        global overSpeed
        overSpeed = threshold  # threshold 값으로 overSpeed 업데이트
        print(f"overSpeed 값이 {threshold}로 변경되었습니다.")  # 변경된 값 출력
        return jsonify({"message": "임계값 업데이트 성공", "threshold": overSpeed}), 200
    else:
        return jsonify({"error": "임계값을 찾을 수 없습니다."}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', threaded=True, port=5001)  # React 앱과 포트 충돌 피하려면 다른 포트 사용
