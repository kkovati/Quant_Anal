import requests


def get_request(url):
    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(f'HTTP Response Status Code: {response.status_code}')
    data_json = requests.get(url).json()

    return data_json


if __name__ == "__main__":
    "Tests"
    json = get_request("http://echo.jsontest.com/key/value/one/two")

    assert json['key'] == 'value'
    assert json['one'] == 'two'

    print("Test1 OK")

    try:
        get_request("http://google.com")
    except Exception as e:
        assert str(type(e)) == "<class 'json.decoder.JSONDecodeError'>"
        print("Test2 OK")
    else:
        raise Exception("Test2 Failed")

    try:
        get_request("http://notworking.notworking")
    except Exception as e:
        assert str(type(e)) == "<class 'requests.exceptions.ConnectionError'>"
        print("Test3 OK")
    else:
        raise Exception("Test3 Failed")
