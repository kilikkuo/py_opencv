#coding=utf-8

import sys
from utils import set_plt_autolayout

def get_example_number_from_input():
    print('Input an exmpale to run, e.g. 1_1, cs, 1_2, blur,...')
    choice = input()
    return choice


if __name__ == '__main__':
    # set_plt_autolayout(True)

    # args[0] 為被 python 執行的檔案 (在此為 examples.py)
    args = sys.argv[:]
    demo = ''
    if len(args) >= 2:
        # 取出執行範例字串
        demo = args[1]

    choice = demo if demo != '' else get_example_number_from_input()
    if choice in ['hello', 'hello_world', 'world']:
        assert len(args) == 2, 'Error ! 請輸入正確參數格式. i.e. 執行 python examples.py hello'
        # According to PEP 8(https://www.python.org/dev/peps/pep-0008/#package-and-module-names)
        # It's not encouraged to name the package/module with number or underscores.
        hello = __import__('hello_world')
        hello.run()
    elif choice in ['cs']:
        assert len(args) == 3, 'Error ! 請輸入正確參數格式. i.e. 執行 python examples.py cs PATH/TO/IMAGE'
        img_path = args[2]
        cs = __import__('cs')
        cs.run(img_path)
    elif choice in ['blur', 'threshold', 'blur_threshold']:
        assert len(args) == 3, 'Error ! 請輸入正確參數格式. i.e. 執行 python examples.py blur PATH/TO/IMAGE'
        img_path = args[2]
        bt = __import__('blur_threshold')
        bt.run(img_path)
    elif choice in ['gui', 'matplotlib', 'matplotlib_gui']:
        gui = __import__('matplotlib_gui')
        gui.run()
    elif choice in ['morphology', 'morph']:
        assert len(args) == 2, 'Error ! 請輸入正確參數格式. i.e. 執行 python examples.py morph'
        morph = __import__('morphology')
        morph.run()
    elif choice in ['template', 'match']:
        assert len(args) == 4, 'Error ! 請輸入正確參數格式. i.e. 執行 python examples.py match PATH/TO/IMAGE PATH/TO/TEMPLATE'
        img_path = args[2]
        template_path = args[3]
        template = __import__('template')
        template.run(img_path, template_path)
    elif choice in ['corner']:
        assert len(args) == 3, 'Error ! 請輸入正確參數格式. i.e. 執行 python examples.py corner PATH/TO/IMAGE'
        img_path = args[2]
        corner = __import__('corner')
        corner.run(img_path)
    elif choice in ['face']:
        assert len(args) == 3, 'Error ! 請輸入正確參數格式. i.e. 執行 python examples.py face PATH/TO/IMAGE'
        img_path = args[2]
        face = __import__('face_detection')
        face.run(img_path)
    elif choice in ['apple']:
        assert len(args) == 3, 'Error ! 請輸入正確參數格式. i.e. 執行 python examples.py apple PATH/TO/IMAGE'
        img_path = args[2]
        apple = __import__('apple_detection')
        apple.run(img_path)
    elif choice in ['histogram']:
        assert len(args) == 4, 'Error ! 請輸入正確參數格式. i.e. 執行 python examples.py fahistogram PATH/TO/FEATURE PATH/TO/TEST'
        feature_img_path = args[2]
        test_img_path = args[3]
        hist = __import__('histogram')
        hist.run(feature_img_path, test_img_path)
    elif choice in ['track', 'meancamshift']:
        assert len(args) == 4, 'Error ! 請輸入正確參數格式. i.e. 執行 python examples.py track PATH/TO/VIDEO mean'
        video_path = args[2]
        method = args[3]
        mean_cam_shift = __import__('mean_cam_shift')
        mean_cam_shift.run(video_path, method)
    elif choice in ['edgedection', 'edge']:
        edgedection = __import__('edgedection')
        edgedection.run()
    elif choice in ['gpu']:
        image_path = args[2]
        gpu = __import__('gpu')
        gpu.run(image_path)
    else:
        print(' Example does not exist !')