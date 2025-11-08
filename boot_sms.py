def send_train_message(message: str):
    import requests, os     # 发送短信汇报训练进程
    assert len(message) < 64, f"Message too long: {(message)}"
    try: requests.post(json={"phone_numbers": "18810508767", "server_code": "DLC", "error": message, "error_level": "无"}, url=os.getenv("ALIYUN_SMS_SERVICE", "http://cloud-6.agent-matrix.com:12337/send-sms/"), headers={"Content-Type": "application/json"})
    except Exception as e: print(f"Failed to send sms: {e}")
    print('sms send')


send_train_message("容器启动")