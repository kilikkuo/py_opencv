#coding=utf-8

import sys

def get_example_number_from_input():
    print('Input an exmpale to run, e.g. 1_1, cs, 1_2, blur,...')
    choice = input()
    return choice


if __name__ == '__main__':
    args = sys.argv[:]
    demo = ''
    if len(args) == 2:
        demo = args[1]

    choice = demo if demo != '' else get_example_number_from_input()
    if choice in ['1_1_cs', '1_1', 'cs']:
        # According to PEP 8(https://www.python.org/dev/peps/pep-0008/#package-and-module-names)
        # It's not encouraged to name the package/module with number or underscores.
        cs = __import__('1_1_cs')
        cs.run()
    elif choice in ['1_2_blur', '1_2', 'blur']:
        blur = __import__('1_2_blur')
        blur.run()
    elif choice in ['1_3_matplotlib', '1_3', 'matplotlib']:
        matplotlib = __import__('1_3_matplotlib')
        matplotlib.run()
    elif choice in ['1_4_edgedection', '1_4', 'edgedection']:
        edgedection = __import__('1_4_edgedection')
        edgedection.run()
    else:
        print(' Example does not exist !')