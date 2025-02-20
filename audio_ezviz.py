from pyezviz import EzvizClient, EzvizCamera
import sys

def main():
    client = EzvizClient("220701263@rajalakshmi.edu.in", "Victory100%", "us")
    try:
        client.login()
        camera = EzvizCamera(client, "BD3102566")
        print(camera.status())
        camera.move('left')
        camera.move('left')
        camera.move('up')
        camera.move('down')
        camera.move('up')
        camera.move('down')
        print("Camera loaded")
    except BaseException as exp:
        print(exp)
        return 1
    finally:
        client.close_session()
if __name__ == '__main__':
    sys.exit(main())