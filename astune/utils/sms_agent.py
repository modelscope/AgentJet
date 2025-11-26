def send_train_message(message: str):
    pass

    # import requests, os, dotenv

    # dotenv.load_dotenv()
    # phone_numbers = os.getenv("PHONE_NUMBERS", None)
    # print("trying to send sms to:", phone_numbers)
    # if phone_numbers:
    #     try:
    #         assert (
    #             len(message) < 15
    #         ), f"message is limit to 15 characters! Current length: {len(message)}."
    #         requests.post(
    #             json={
    #                 "phone_numbers": phone_numbers,
    #                 "server_code": "DLC",
    #                 "error": message,
    #                 "error_level": "wbb1u1g0dte2n",
    #             },
    #             url=os.getenv(
    #                 "ALIYUN_SMS_SERVICE",
    #                 "http://cloud-6.agent-matrix.com:12337/send-sms/",
    #             ),
    #             headers={"Content-Type": "application/json"},
    #         )
    #     except Exception as e:
    #         print(f"Failed to send sms: {e}")
