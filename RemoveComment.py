# -*- coding: utf-8 -*-

import os, sys
import shutil

class CodeMode:
    plaintext = 0   
    single = 1      
    multiple = 2    
    string = 3      
    char = 4        

code_exts = ['h', 'c', 'cpp', 'java']

def create_dir(dir_path):
    try:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        if not os.path.exists(dir_path):
            print("[-] Create directory {0} failed".format(dir_path))
            return False
        if os.path.isfile(dir_path):
            print("{0} is file".format(dir_path))
            return False
        else:
            return True
    except BaseException as e:
        print("[-] Create directory {0} failed: {1}".format(dir_path, e))
    return False


def DeleteSpaceLine(file_path):
    if (not os.path.exists(file_path)):
        print('[-] 输入路径{0}不存在'.format(os.path.abspath(file_path)))
        return
    if (not os.path.isfile(file_path)):
        print('[-] 输入路径{0}不是文件'.format(os.path.abspath(file_path)))
        return
    print('\tDelete extra blank lines in file')
    file = open(file_path, 'r')
    lines = file.readlines()
    file.close()
    file = open(file_path, 'w')
    last_line_blank = False
    for line in lines:
        if (len(line) == (line.count(' ') + line.count('\t') + line.count('\r') + line.count('\n'))):
            if (not last_line_blank):
                file.writelines(line)
            last_line_blank = True
        else:
            file.writelines(line)
            last_line_blank = False
    file.close()


def DeleteSpaceLineUtf8(file_path):
    if (not os.path.exists(file_path)):
        print('[-] 输入路径{0}不存在'.format(os.path.abspath(file_path)))
        return
    if (not os.path.isfile(file_path)):
        print('[-] 输入路径{0}不是文件'.format(os.path.abspath(file_path)))
        return
    print('\tDelete extra blank lines in file')
    file = open(file_path, 'r', encoding='utf-8')
    lines = file.readlines()
    file.close()
    file = open(file_path, 'w', encoding='utf-8')
    last_line_blank = False
    for line in lines:
        if (len(line) == (line.count(' ') + line.count('\t') + line.count('\r') + line.count('\n'))):
            if (not last_line_blank):
                file.writelines(line)
            last_line_blank = True
        else:
            file.writelines(line)
            last_line_blank = False
    file.close()


def RemoveComment(in_file_path, out_file_path):
    if (not os.path.exists(in_file_path)):
        print('[-] 输入路径{0}不存在'.format(os.path.abspath(in_file_path)))
        return
    if (not os.path.isfile(in_file_path)):
        print('[-] 输入路径{0}不是文件'.format(os.path.abspath(in_file_path)))
        return

    abs_in_file_path = os.path.abspath(in_file_path)
    abs_out_file_path = os.path.abspath(out_file_path)

    print('[*] 输入文件{0}'.format(abs_in_file_path))
    print('[*] 输出文件{0}'.format(abs_out_file_path))
    if (abs_out_file_path == abs_in_file_path):
        print('[*] 输入输出路径相同，忽略')
        return

    with open(in_file_path, 'r') as fin:
        with open(out_file_path, 'w') as fout:
            RealRemoveComment(fin, fout)
    DeleteSpaceLine(out_file_path)

def RemoveCommentUtf8(in_file_path, out_file_path):
    if (not os.path.exists(in_file_path)):
        print('[-] 输入路径{0}不存在'.format(os.path.abspath(in_file_path)))
        return
    if (not os.path.isfile(in_file_path)):
        print('[-] 输入路径{0}不是文件'.format(os.path.abspath(in_file_path)))
        return

    abs_in_file_path = os.path.abspath(in_file_path)
    abs_out_file_path = os.path.abspath(out_file_path)

    print('[*] 输入文件{0}'.format(abs_in_file_path))
    print('[*] 输出文件{0}'.format(abs_out_file_path))
    if (abs_out_file_path == abs_in_file_path):
        print('[*] 输入输出路径相同，忽略')
        return
    
    with open(in_file_path, 'r', encoding='utf-8') as fin:
        with open(out_file_path, 'w', encoding='utf-8') as fout:
            RealRemoveComment(fin, fout)
    DeleteSpaceLineUtf8(out_file_path)


def RealRemoveComment(fin, fout):
    mode = CodeMode.plaintext
    last = ''
    last_last = ''
    first_line_in_mutiple = True
    space_before_mutiple = False
    after_mutiple = False

    line_count = 1

    while 1:
        current = fin.read(1)
        if (not current):
            print('\r\tProcessed {0} lines'.format(line_count))
            break
        if (current == '\n'):
            print('\r\tProcessed {0} lines'.format(line_count), end = '')
            line_count = line_count + 1
        if (CodeMode.single == mode):
            if (last != '\\' and (current == '\n' or current == '\r')):
                fout.write(current)
                current = '\0'
                mode = CodeMode.plaintext
        elif (CodeMode.multiple == mode):
            if (last == '*' and current == '/'):
                current = '\0'
                mode = CodeMode.plaintext
                if (first_line_in_mutiple == True):
                    if (space_before_mutiple == True):
                        after_mutiple = False
                    else:
                        after_mutiple = True
                else:
                    after_mutiple = False
            elif (current == '\n' or current == '\r'):
                if (first_line_in_mutiple == True):
                    fout.write(current)
                    if (current == '\n'):
                        first_line_in_mutiple = False
        elif (CodeMode.string == mode):
            if (last == '\\'):
                fout.write(last)
                fout.write(current)
                current = '\0'
            elif (current != '\\'):
                fout.write(current)
                if (current == '"'):
                    mode = CodeMode.plaintext
        elif (CodeMode.char == mode):
            if (last == '\\'):
                fout.write(last)
                fout.write(current)
                current = '\0'
            elif (current != '\\'):
                fout.write(current)
                if (current == "'"):
                    mode = CodeMode.plaintext
        else:
            if (last == '/'):
                if (current == '/'):
                    mode = CodeMode.single
                    current = '\0'
                elif (current == '*'):
                    mode = CodeMode.multiple
                    current = '\0'
                    first_line_in_mutiple = True
                    if (last_last == ' ' or last_last == '\n' or last_last == '\t'):
                        space_before_mutiple = True
                    else:
                        space_before_mutiple = False
                else:
                    if (after_mutiple == True):
                        fout.write(' ')
                        after_mutiple = False
                    fout.write(last)
                    fout.write(current)
            elif (current != '/'):
                if (after_mutiple == True):
                    if (current != ' ' and current != '\r' and current != '\n' and current != '\t'):
                        fout.write(' ')
                    after_mutiple = False
                fout.write(current)
                if (current == '"'):
                    mode = CodeMode.string
                elif (current == "'"):
                    mode = CodeMode.char
        last_last = last
        last = current


def RemoveCommentOutFolder(in_file_path, out_dir_path):
    abs_in_file_path = os.path.abspath(in_file_path)
    abs_out_dir_path = os.path.abspath(out_dir_path)
    if (not os.path.exists(in_file_path)):
        print('[-] 输入路径{0}不存在'.format(abs_in_file_path))
        return
    if (not os.path.isfile(in_file_path)):
        print('[-] 输入路径{0}不是文件'.format(abs_in_file_path))
        return
    if (not os.path.exists(out_dir_path)):
        create_dir(out_dir_path)
    if (not os.path.exists(out_dir_path)):
        print('[-] 无法创建输出文件夹{0}'.format(abs_out_dir_path))
        return
    if (not os.path.isdir(out_dir_path)):
        print('[-] 输出路径{0}不是文件夹'.format(abs_out_dir_path))
        return
    in_file_dir, in_file_name = os.path.split(abs_in_file_path)
    out_file_path = os.path.join(abs_out_dir_path, in_file_name)
    RemoveComment(abs_in_file_path, out_file_path)


def RemoveAllCommentInFolder(in_dir_path, out_dir_path):
    abs_in_dir_path = os.path.abspath(in_dir_path)
    abs_out_dir_path = os.path.abspath(out_dir_path)
    if (not os.path.exists(in_dir_path)):
        print('[-] 输入路径{0}不存在'.format(abs_in_dir_path))
        return
    if (not os.path.isdir(in_dir_path)):
        print('[-] 输入路径{0}不是文件夹'.format(abs_in_dir_path))
        return
    if (os.path.exists(out_dir_path)):
        if (not os.path.isdir(out_dir_path)):
            print('[-] 输出路径{0}不是文件夹'.format(abs_out_dir_path))
            return
    else:
        create_dir(out_dir_path)
        if (not os.path.exists(out_dir_path)):
            print('[-] 无法创建输出文件夹{0}'.format(abs_out_dir_path))
            return
        else:
            if (not os.path.isdir(out_dir_path)):
                print('[-] 无法创建输出文件夹{0}'.format(abs_out_dir_path))
                return

    for in_name in os.listdir(abs_in_dir_path):
        in_path = os.path.join(abs_in_dir_path, in_name)
        out_path = os.path.join(abs_out_dir_path, in_name)
        if (os.path.isfile(in_path)):
            ext = os.path.splitext(in_name)[1][1:].lower()
            if (ext in code_exts):
                #C
                try:
                    RemoveComment(in_path, out_path)
                except Exception as e:
                    print("[-] Remove comment from {0} failed: {1}".format(in_path, e))
                    print("[*] Try Utf8")
                    try:
                        RemoveCommentUtf8(in_path, out_path)
                    except Exception as e1:
                        print("[-] Remove comment from {0} failed: {1}".format(in_path, e1))
                        print('[*] 仅执行复制操作')
                        shutil.copyfile(in_path, out_path)
                        #os.system("pause")
            else:
                print('[*] 文件{0}不是支持的类型，仅复制'.format(in_path))
                shutil.copyfile(in_path, out_path)
        else:
            RemoveAllCommentInFolder(in_path, out_path)



def Exit(code):
    os.system("pause")
    exit(code)

if __name__ == '__main__':
    argv = sys.argv
    if(len(argv) != 3):
        print("[-] 需要输入路径和输出路径作为参数!")
        Exit(-1)

    try:
        if (os.path.exists(argv[1])):
            if (os.path.isdir(argv[1])):
                RemoveAllCommentInFolder(argv[1], argv[2])
            else:
                if (os.path.exists(argv[2])):
                    if (os.path.isdir(argv[2])):
                        RemoveCommentOutFolder(argv[1], argv[2])
                    else:
                        RemoveComment(argv[1], argv[2])
                else:
                    RemoveComment(argv[1], argv[2])
        else:
            print("[-] 输入路径{0}不存在!".format(argv[1]))
        print('[*] 处理完成')
    except Exception as e:
        print("[-] Remove comment from {0} failed: {1}".format(argv[1], e))
        print('[-] 处理过程中出现错误')
